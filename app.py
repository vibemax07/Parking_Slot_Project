"""
Smart Parking System — Web Frontend
────────────────────────────────────
Flask server that provides a beautiful web UI for the parking visualizer.
Run:  python app.py
Then open:  http://localhost:5000
"""

from flask import Flask, render_template, jsonify, send_file
import cv2
import numpy as np
import base64
import os
import math

# ── Import parking logic from the existing script ────────────────────────────
from Car_Animation import (
    load_vehicle_image, detect_slots, find_best_slot,
    build_base_frame, build_path, build_exit_path, interpolate_path,
    draw_car, compute_car_dimensions,
    CAR_IMAGE_BASE_ANGLES, CAR_SIZES, CAR_FILL_MULTIPLIERS,
    MODEL_PATH, IMAGE_PATH, SLOTS_PKL_PATH, VEHICLES_DIR,
    EXTRA_ROAD_H, _draw_extra_road, _draw_best_slot_glow,
)

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  CACHED ASSETS  (loaded once on first request)
# ══════════════════════════════════════════════════════════════════════════════
_cache = {}

# ── Persistent state: track parked cars across requests ──────────────────────
# Each entry: { car_type, vehicle_rgba, cx, cy, angle, car_len, car_wid,
#               slot_index, slot_cx, slot_cy, slot_x, slot_y, slot_w, slot_h }
_parked_cars = []


def get_assets():
    """Load & cache model + image + slots so we don't reload every request."""
    if "model" not in _cache:
        from tensorflow.keras.models import load_model
        print("  [*] Loading TensorFlow model (first request)...")
        _cache["model"] = load_model(MODEL_PATH)
        _cache["img"]   = cv2.imread(IMAGE_PATH)
        import pickle
        with open(SLOTS_PKL_PATH, "rb") as f:
            _cache["slots"] = pickle.load(f)
        _cache["slots_info"] = detect_slots(
            _cache["model"], _cache["img"], _cache["slots"]
        )
        print("  [OK] Model and assets cached.")
    return (_cache["model"], _cache["img"],
            _cache["slots"], _cache["slots_info"])


def _draw_all_parked_cars(frame):
    """Draw every previously parked car onto the given frame."""
    for pc in _parked_cars:
        draw_car(frame, pc["cx"], pc["cy"], pc["angle"],
                 pc["car_type"], pc["vehicle_rgba"],
                 pc["car_len"], pc["car_wid"])


def _build_reference_image(img, slots_info):
    """Build the reference image with slot overlays + parked cars drawn + extra road."""
    # Color constants — kept in sync with Car_Animation.py
    EMPTY_COLOR = (0, 255, 0)   # rgb(109,238,109) in BGR (same values, symmetric)
    OCC_COLOR   = (0,  0,  255)   # red for occupied slots

    orig_h, W = img.shape[:2]
    extra = np.zeros((EXTRA_ROAD_H, W, 3), dtype=np.uint8)
    frame   = np.vstack([img.copy(), extra])
    overlay = frame.copy()
    for slot in slots_info:
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        colour = EMPTY_COLOR if slot["status"] == "Empty" else OCC_COLOR
        cv2.rectangle(overlay, (x, y), (x+w, y+h), colour, -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    for i, slot in enumerate(slots_info):
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        colour = EMPTY_COLOR if slot["status"] == "Empty" else OCC_COLOR
        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 1)
        cv2.putText(frame, str(i+1), (x+3, y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

    # Draw extra road strip
    _draw_extra_road(frame, orig_h)

    # Draw all previously parked cars on the reference image
    _draw_all_parked_cars(frame)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/vehicle-image/<car_type>")
def vehicle_image(car_type):
    """Serve the top-down vehicle PNG for preview cards."""
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        path = os.path.join(VEHICLES_DIR, car_type + ext)
        if os.path.isfile(path):
            return send_file(path, mimetype="image/png")
    return "", 404


@app.route("/api/reference")
def reference():
    """Return the reference image with coloured slot overlays + stats."""
    _, img, _, slots_info = get_assets()

    frame = _build_reference_image(img, slots_info)

    # Resize to 960 wide, preserve aspect ratio (includes extra road height)
    h_orig, w_orig = frame.shape[:2]
    out_w = 960
    out_h = int(h_orig * out_w / w_orig)
    small = cv2.resize(frame, (out_w, out_h))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64   = base64.b64encode(buf).decode("utf-8")

    empty    = sum(1 for s in slots_info if s["status"] == "Empty")
    occupied = len(slots_info) - empty

    return jsonify(image=b64,
                   stats=dict(total=len(slots_info),
                              empty=empty, occupied=occupied))


@app.route("/api/park/<car_type>")
def park(car_type):
    """
    Run the full parking pipeline for *car_type* and return:
      • animation frames  (base64-encoded JPEGs, sampled for the web)
      • parking metadata   (slot number, stats, etc.)
    After parking, the slot is marked as Occupied so the next car
    picks a different slot.
    """
    if car_type not in CAR_SIZES:
        return jsonify(error=f"Unknown vehicle type: {car_type}"), 400

    _, img, _, slots_info = get_assets()

    # Check if there are any empty slots left
    empty_count = sum(1 for s in slots_info if s["status"] == "Empty")
    if empty_count == 0:
        return jsonify(error="Parking lot is FULL! No empty slots available.",
                       lot_full=True), 404

    vehicle_rgba = load_vehicle_image(car_type)

    best_slot, _ = find_best_slot(slots_info, car_type, img.shape)
    if best_slot is None:
        return jsonify(error=f"No suitable slot for {car_type.upper()}. "
                       f"Try a smaller vehicle."), 404

    # ── Slot geometry ────────────────────────────────────────────────────
    slot_x, slot_y = best_slot["x"], best_slot["y"]
    slot_w, slot_h = best_slot["w"], best_slot["h"]
    slot_num = str(best_slot["index"] + 1)

    slot_cx = slot_x + slot_w // 2
    slot_cy = slot_y + slot_h // 2

    # ── Path + base frame ────────────────────────────────────────────────
    waypoints = build_path(img.shape, slot_cx, slot_cy,
                           slot_x, slot_y, slot_w, slot_h, car_type)
    base = build_base_frame(img, slots_info, best_slot, None)

    # Draw all previously parked cars onto the base frame
    _draw_all_parked_cars(base)

    final_angle = waypoints[-1][2]
    car_len, car_wid = compute_car_dimensions(
        vehicle_rgba, slot_w, slot_h, final_angle, car_type, fill_factor=1.32
    )

    # ── Generate animation frames ────────────────────────────────────────
    interp = interpolate_path(waypoints, steps=50)
    trail  = []
    raw_frames = []

    # Driving phase
    for fi, (fx, fy, fa) in enumerate(interp):
        frame = base.copy()
        trail.append((fx, fy))
        if len(trail) > 1:
            for ti in range(1, len(trail)):
                if ti % 8 < 4:
                    cv2.line(frame, trail[ti-1], trail[ti], (0, 190, 210), 1)
        draw_car(frame, fx, fy, fa, car_type, vehicle_rgba, car_len, car_wid)
        pct = int(fi / max(len(interp)-1, 1) * 100)
        cv2.rectangle(frame, (8, 6), (310, 48), (15, 15, 15), -1)
        cv2.putText(frame, f"Navigating...  {pct}%", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 215, 255), 2)
        raw_frames.append(frame)

    # Parking fade-in phase
    fx, fy, fa = interp[-1]
    for step in range(20):
        a = step / 19.0
        frame = base.copy()
        layer = frame.copy()
        draw_car(layer, fx, fy, fa, car_type, vehicle_rgba, car_len, car_wid)
        cv2.addWeighted(layer, a, frame, 1 - a, 0, frame)
        cv2.rectangle(frame, (8, 6), (230, 48), (15, 15, 15), -1)
        cv2.putText(frame, "Parking...", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (50, 255, 120), 2)
        raw_frames.append(frame)

    # Final "parked" frame
    final = base.copy()
    draw_car(final, fx, fy, fa, car_type, vehicle_rgba, car_len, car_wid)
    cv2.rectangle(final, (8, 6), (570, 48), (15, 15, 15), -1)
    cv2.putText(final,
                f"{car_type.upper()} parked in Slot {slot_num}!",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (50, 255, 120), 2)
    raw_frames.append(final)

    # ══════════════════════════════════════════════════════════════════════
    #  MARK SLOT AS OCCUPIED  — so the next car picks a different slot
    # ══════════════════════════════════════════════════════════════════════
    best_slot["status"] = "Occupied"

    # Save this parked car so it appears on all future frames
    _parked_cars.append({
        "car_type": car_type,
        "vehicle_rgba": vehicle_rgba,
        "cx": int(fx),
        "cy": int(fy),
        "angle": fa,
        "car_len": car_len,
        "car_wid": car_wid,
        # Store slot geometry so we can reconstruct the exit path on demand
        "slot_index": best_slot["index"],
        "slot_cx": slot_cx,
        "slot_cy": slot_cy,
        "slot_x":  slot_x,
        "slot_y":  slot_y,
        "slot_w":  slot_w,
        "slot_h":  slot_h,
    })

    # ── Sample & encode for web ──────────────────────────────────────────
    step = 3                       # every 3rd frame ≈ 10 FPS feel
    indices = list(range(0, len(raw_frames), step))
    # Always include the last 8 frames (parking phase + final)
    tail = list(range(max(0, len(raw_frames) - 8), len(raw_frames)))
    indices = sorted(set(indices + tail))

    encoded = []
    for idx in indices:
        f = raw_frames[idx]
        # Resize to 960 wide, preserve aspect ratio (extra road included)
        fh, fw = f.shape[:2]
        out_w = 960
        out_h = int(fh * out_w / fw)
        small = cv2.resize(f, (out_w, out_h))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        encoded.append(base64.b64encode(buf).decode("utf-8"))

    # Updated counts AFTER marking the slot occupied
    empty = sum(1 for s in slots_info if s["status"] == "Empty")

    return jsonify(
        frames=encoded,
        slot_number=slot_num,
        slot_index=best_slot["index"],   # 0-based, for exit matching in history
        car_type=car_type,
        stats=dict(total=len(slots_info), empty=empty,
                   occupied=len(slots_info) - empty),
    )


@app.route("/api/parked-cars")
def parked_cars():
    """
    Return the bounding boxes (in 960-wide display coords) of every parked car,
    so the frontend can do click-hit-testing.
    """
    _, img, _, _ = get_assets()
    orig_h, orig_w = img.shape[:2]
    total_h = orig_h + EXTRA_ROAD_H

    # Scale factor: we display at 960 wide
    scale = 960 / orig_w
    out_h = int(total_h * scale)

    result = []
    for idx, pc in enumerate(_parked_cars):
        hw = int(pc["car_len"] * scale / 2) + 4   # small padding
        hh = int(pc["car_wid"] * scale / 2) + 4
        cx = int(pc["cx"] * scale)
        cy = int(pc["cy"] * scale)
        result.append({
            "id": idx,
            "car_type": pc["car_type"],
            "cx": cx, "cy": cy,
            "hw": hw, "hh": hh,
        })

    return jsonify(cars=result, display_w=960, display_h=out_h)


@app.route("/api/exit/<int:car_id>")
def exit_car(car_id):
    """
    Animate a parked car exiting its slot along the CORRECT road path,
    then mark its slot as empty again.
    """
    _, img, _, slots_info = get_assets()

    if car_id < 0 or car_id >= len(_parked_cars):
        return jsonify(error="Invalid car id"), 400

    pc = _parked_cars[car_id]
    car_type    = pc["car_type"]
    vehicle_rgba = pc["vehicle_rgba"]
    car_len     = pc["car_len"]
    car_wid     = pc["car_wid"]
    slot_cx     = pc["slot_cx"]
    slot_cy     = pc["slot_cy"]
    slot_x      = pc["slot_x"]
    slot_y      = pc["slot_y"]
    slot_w      = pc["slot_w"]
    slot_h      = pc["slot_h"]
    slot_index  = pc["slot_index"]

    # Build the exit path (reversed entry path)
    waypoints = build_exit_path(img.shape, slot_cx, slot_cy,
                                slot_x, slot_y, slot_w, slot_h, car_type)

    # Remove this car from the list BEFORE building the base frame,
    # so the base frame shows the lot without this car
    _parked_cars.pop(car_id)

    # Mark the slot as empty again
    for slot in slots_info:
        if slot["index"] == slot_index:
            slot["status"] = "Empty"
            break

    # Build a base frame WITHOUT this car in it
    base = build_base_frame(img, slots_info, None, None)
    _draw_all_parked_cars(base)   # draw remaining cars

    # Compute final angle (parked pose) — this is the START of the exit
    final_park_angle = waypoints[0][2]

    # Generate exit animation frames
    interp = interpolate_path(waypoints, steps=50)
    trail  = []
    raw_frames = []

    for fi, (fx, fy, fa) in enumerate(interp):
        frame = base.copy()
        trail.append((fx, fy))
        if len(trail) > 1:
            for ti in range(1, len(trail)):
                if ti % 8 < 4:
                    cv2.line(frame, trail[ti-1], trail[ti], (255, 140, 0), 1)
        draw_car(frame, fx, fy, fa, car_type, vehicle_rgba, car_len, car_wid)
        pct = int(fi / max(len(interp)-1, 1) * 100)
        cv2.rectangle(frame, (8, 6), (330, 48), (15, 15, 15), -1)
        cv2.putText(frame, f"Exiting...  {pct}%", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 160, 50), 2)
        raw_frames.append(frame)

    # Final frame — car is gone, slot is free
    final_frame = base.copy()
    cv2.rectangle(final_frame, (8, 6), (440, 48), (15, 15, 15), -1)
    cv2.putText(final_frame, f"{car_type.upper()} exited. Slot is now free!",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 200, 255), 2)
    raw_frames.append(final_frame)

    # Sample & encode frames
    step = 3
    indices = list(range(0, len(raw_frames), step))
    tail = list(range(max(0, len(raw_frames) - 8), len(raw_frames)))
    indices = sorted(set(indices + tail))

    encoded = []
    for idx2 in indices:
        f = raw_frames[idx2]
        fh, fw = f.shape[:2]
        out_w = 960
        out_h = int(fh * out_w / fw)
        small = cv2.resize(f, (out_w, out_h))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        encoded.append(base64.b64encode(buf).decode("utf-8"))

    empty = sum(1 for s in slots_info if s["status"] == "Empty")

    return jsonify(
        frames=encoded,
        car_type=car_type,
        slot_index=slot_index,
        stats=dict(total=len(slots_info), empty=empty,
                   occupied=len(slots_info) - empty),
    )


@app.route("/api/reset")
def reset():
    """Reset all parked cars — re-detect slots from scratch."""
    global _parked_cars
    _parked_cars.clear()

    # Force re-detection of slots
    if "model" in _cache:
        _cache["slots_info"] = detect_slots(
            _cache["model"], _cache["img"], _cache["slots"]
        )

    _, img, _, slots_info = get_assets()
    empty = sum(1 for s in slots_info if s["status"] == "Empty")

    return jsonify(
        success=True,
        stats=dict(total=len(slots_info), empty=empty,
                   occupied=len(slots_info) - empty),
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(VEHICLES_DIR, exist_ok=True)
    print("\n+==========================================+")
    print("|   Smart Parking System  -- Web UI        |")
    print("|   Open  http://localhost:5000             |")
    print("+==========================================+\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
