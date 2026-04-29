"""
parking_visualizer.py  —  v6
────────────────────────────
Uses REAL car images (PNG/JPG with transparent or white background)
as sprites. Drop your top-down car images into the vehicles/ folder
and name them:

    vehicles/
        sedan.png
        hatchback.png
        suv.png
        truck.png

The script loads the image, intelligently removes its background, rotates
it to follow the road, and overlays it onto the parking lot as the car
drives to its best slot — using the ACTUAL road layout from Reference.jpg.

HOW TO USE:
  1. Create a folder called  vehicles/  next to this script.
  2. Drop top-down car PNGs in it (name = vehicle type, e.g. suv.png).
     Use images with a WHITE or transparent background for best results.
  3. Run:  python Car_Animation.py

If no image is found for a type it falls back to the drawn sprite.
"""

import cv2
import numpy as np
import pickle
import math
import sys
import os

MODEL_PATH     = r"C:\Users\DELL\OneDrive\Documents\java final prep\Parking_Slot_Project\parking_model.h5"
IMAGE_PATH     = r"C:\Users\DELL\OneDrive\Documents\java final prep\Parking_Slot_Project\Reference.jpg"
SLOTS_PKL_PATH = r"C:\Users\DELL\OneDrive\Documents\java final prep\Parking_Slot_Project\slots.pkl"

# Folder that holds your vehicle images (relative to this script)
VEHICLES_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicles")

ENTRY_POINT = (0, 490)     # left edge of the lower (right-bound) lane

CAR_SIZES = {
    "hatchback": 1500,
    "sedan":     2500,
    "suv":       3500,
}

# ── Per-vehicle base angles ──────────────────────────────────────────────────
# The heading (in degrees) that the car faces IN ITS IMAGE FILE.
#   0 = right,  90 = down,  180 = left,  270 = up
# Adjust these if your replacement image points a different way.
CAR_IMAGE_BASE_ANGLES = {
    "sedan":     0,     # front faces RIGHT
    "hatchback": 180,   # front faces LEFT in the source image file
    "suv":       0,     # front faces RIGHT
}

# ── Per-vehicle fill factors (relative to slot size) ─────────────────────────
# Multiplied against the base fill_factor in compute_car_dimensions.
# Values > 1.0 make the car larger within the slot.
CAR_FILL_MULTIPLIERS = {
    "sedan":     1.00,  # unchanged
    "hatchback": 1.40,  # +40% larger
    "suv":       1.25,  # +25% larger
}

# Fallback drawn-sprite colours when no image file is found
CAR_PALETTE = {
    "sedan":     ((30,  10,  140), (60,  20,  210), (120, 60, 255),  (10,  5,   60)),
    "hatchback": ((10,  100, 180), (20,  160, 255), (120, 210, 255), (5,   50,  90)),
    "suv":       ((10,  80,  10),  (20,  140, 20),  (100, 220, 100), (5,   40,  5)),
}


# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLE IMAGE LOADER  (smart, edge-aware background removal)
# ══════════════════════════════════════════════════════════════════════════════

def _smart_remove_background(bgr_img):
    """
    Remove the background of a car image using an edge-aware approach:

      1. Detect edges (Canny) to find the car outline.
      2. Dilate edges to create strong boundary lines.
      3. Mark all bright/light pixels (B,G,R > 200) as "potential background".
      4. SUBTRACT the dilated edges so the car boundary breaks any connection
         between the vehicle body and the surrounding background.
      5. Connected-component analysis — only components that touch an image
         border are real background.  Interior light regions (e.g. a white
         car roof, a cream truck bed) stay opaque.
      6. Expand the final background mask slightly and apply as alpha.

    This correctly handles:
      - White cars on white/gray backgrounds  (hatchback, sedan)
      - Light-coloured cargo beds             (truck)
      - Dark cars on white backgrounds        (SUV)
      - Images with watermarks                (stock images)
    """
    h, w = bgr_img.shape[:2]

    # ── 1. Edge detection ────────────────────────────────────────────────
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 80)
    # Thick edges to reliably seal the car boundary
    edge_wall = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)

    # ── 2. Potential-background mask (bright pixels) ─────────────────────
    b, g, r = cv2.split(bgr_img)
    pot_bg = ((b > 200) & (g > 200) & (r > 200)).astype(np.uint8) * 255
    # Break connections at the car edges
    pot_bg[edge_wall > 0] = 0

    # ── 3. Connected components — keep only border-touching ones ─────────
    num_labels, labels = cv2.connectedComponents(pot_bg)

    border_labels = set()
    border_labels.update(labels[0,  :].ravel())     # top row
    border_labels.update(labels[-1, :].ravel())     # bottom row
    border_labels.update(labels[:,  0].ravel())     # left column
    border_labels.update(labels[:, -1].ravel())     # right column
    border_labels.discard(0)                        # label 0 = non-pot-bg

    bg_mask = np.zeros((h, w), np.uint8)
    for lbl in border_labels:
        bg_mask[labels == lbl] = 255

    # ── 4. Expand to catch fringe pixels and smooth ──────────────────────
    bg_mask = cv2.dilate(bg_mask, np.ones((5, 5), np.uint8), iterations=2)
    bg_mask = cv2.GaussianBlur(bg_mask, (3, 3), 0)

    alpha = np.where(bg_mask > 128, 0, 255).astype(np.uint8)

    # Close small holes inside the car silhouette
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE,
                             np.ones((7, 7), np.uint8), iterations=2)

    return alpha


def load_vehicle_image(car_type):
    """
    Try to load vehicles/<car_type>.png  (or .jpg / .jpeg).
    Returns an RGBA numpy array (H, W, 4) or None if not found.
    """
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    path = None
    for ext in exts:
        candidate = os.path.join(VEHICLES_DIR, car_type + ext)
        if os.path.isfile(candidate):
            path = candidate
            break

    if path is None:
        print(f"  [vehicles/{car_type}.png not found -- using drawn sprite]")
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  [Could not read {path} -- using drawn sprite]")
        return None

    # ── Already has an alpha channel ─────────────────────────────────────
    if img.shape[2] == 4:
        # Check if the alpha channel is actually used (not all 255)
        if img[:, :, 3].min() < 250:
            print(f"  [Loaded {path} with existing alpha channel]")
            return img
        # Alpha exists but is fully opaque → treat as no-alpha and continue
        print(f"  [Loaded {path} -- alpha is opaque, will auto-remove background]")
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img

    # ── Smart background removal ─────────────────────────────────────────
    alpha = _smart_remove_background(bgr)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    print(f"  [Loaded {path}, smart background removal applied]")
    return rgba


# ══════════════════════════════════════════════════════════════════════════════
#  ROTATE + OVERLAY  (works with RGBA sprite)
# ══════════════════════════════════════════════════════════════════════════════

def rotate_image(rgba, angle_deg):
    """Rotate an RGBA image around its centre without cropping."""
    h, w = rgba.shape[:2]
    diag = int(math.ceil(math.hypot(w, h)))
    # Pad to diagonal so nothing gets clipped during rotation
    pad_x = (diag - w) // 2
    pad_y = (diag - h) // 2
    padded = cv2.copyMakeBorder(rgba, pad_y, pad_y, pad_x, pad_x,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ph, pw = padded.shape[:2]
    M = cv2.getRotationMatrix2D((pw // 2, ph // 2), -angle_deg, 1.0)
    rotated = cv2.warpAffine(padded, M, (pw, ph),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    return rotated


def overlay_rgba(background, sprite_rgba, cx, cy):
    """
    Alpha-composite sprite_rgba onto background centred at (cx, cy).
    The sprite is used at its current pixel size (no resizing here).
    """
    h, w = sprite_rgba.shape[:2]

    x1 = cx - w // 2;  y1 = cy - h // 2
    x2 = x1 + w;       y2 = y1 + h

    # Clamp to background bounds
    bh, bw = background.shape[:2]
    sx1 = max(0, -x1);  sy1 = max(0, -y1)

    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(bw, x2);  y2c = min(bh, y2)

    if x2c <= x1c or y2c <= y1c:
        return

    roi        = background[y1c:y2c, x1c:x2c].astype(np.float32)
    sprite_roi = sprite_rgba[sy1:sy1+(y2c-y1c), sx1:sx1+(x2c-x1c)]

    if sprite_roi.shape[2] == 4:
        alpha = sprite_roi[:, :, 3:4].astype(np.float32) / 255.0
        bgr   = sprite_roi[:, :, :3].astype(np.float32)
        blended = bgr * alpha + roi * (1.0 - alpha)
    else:
        blended = sprite_roi[:, :, :3].astype(np.float32)

    background[y1c:y2c, x1c:x2c] = np.clip(blended, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  FALLBACK DRAWN SPRITE (used when no image file exists)
# ══════════════════════════════════════════════════════════════════════════════

def _rot(lx, ly, cx, cy, rad):
    rx = lx * math.cos(rad) - ly * math.sin(rad)
    ry = lx * math.sin(rad) + ly * math.cos(rad)
    return (int(cx + rx), int(cy + ry))

def _poly(pts):
    return np.array(pts, dtype=np.int32)

def draw_fallback_car(canvas, cx, cy, angle_deg, car_type, length=90, width=44):
    palette   = CAR_PALETTE.get(car_type, CAR_PALETTE["sedan"])
    body_dark, body_mid, body_hi, roof_dark = palette
    rad = math.radians(angle_deg)
    def r(lx, ly): return _rot(lx, ly, cx, cy, rad)
    L = length // 2;  W = width // 2

    shadow = canvas.copy()
    sh_pts = [r(L*0.95*math.cos(2*math.pi*i/36)+5, W*0.85*math.sin(2*math.pi*i/36)+5)
              for i in range(36)]
    cv2.fillPoly(shadow, [_poly(sh_pts)], (20, 20, 20))
    cv2.addWeighted(shadow, 0.35, canvas, 0.65, 0, canvas)

    body = [r(L*math.cos(a), W*(1-.28*abs(math.cos(a))**1.6)*math.sin(a))
            for a in [2*math.pi*i/72 for i in range(72)]]
    cv2.fillPoly(canvas, [_poly(body)], body_mid)

    hi_pts = ([r(L*math.cos(a), W*(1-.28*abs(math.cos(a))**1.6)*math.sin(a))
               for a in [math.pi*.55+math.pi*.45*i/29 for i in range(30)]] +
              [r(L*.92*math.cos(a), (W-3)*(1-.28*abs(math.cos(a))**1.6)*math.sin(a))
               for a in [math.pi*.55+math.pi*.45*i/29 for i in range(29,-1,-1)]])
    cv2.fillPoly(canvas, [_poly(hi_pts)], body_hi)

    ws_f = _poly([r(L*.62,W*.38), r(L*.30,W*.50), r(L*.30,-W*.50), r(L*.62,-W*.38)])
    cv2.fillPoly(canvas, [ws_f], (25, 25, 25))
    ws_r = _poly([r(-L*.30,W*.46), r(-L*.65,W*.30), r(-L*.65,-W*.30), r(-L*.30,-W*.46)])
    cv2.fillPoly(canvas, [ws_r], (35, 35, 38))
    roof = _poly([r(L*.28,W*.38), r(-L*.28,W*.44), r(-L*.28,-W*.44), r(L*.28,-W*.38)])
    cv2.fillPoly(canvas, [roof], roof_dark)

    for wlx in [L*.55, -L*.55]:
        for wly in [W+3, -(W+3)]:
            wc = r(wlx, wly)
            ta = max(4, int(length*.12)); tb = max(3, int(width*.09))
            cv2.ellipse(canvas, wc, (ta,tb), angle_deg, 0, 360, (18,18,18), -1)
            cv2.ellipse(canvas, wc, (ta-2,tb-1), angle_deg, 0, 360, (180,185,190), -1)

    outline = [r(L*math.cos(a), W*(1-.28*abs(math.cos(a))**1.6)*math.sin(a))
               for a in [2*math.pi*i/72 for i in range(72)]]
    cv2.polylines(canvas, [_poly(outline)], True, (10,10,10), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED DRAW FUNCTION — image sprite OR fallback
# ══════════════════════════════════════════════════════════════════════════════

def compute_car_dimensions(vehicle_rgba, slot_w, slot_h, parking_angle_deg,
                           car_type, fill_factor=0.88):
    """
    Compute the unrotated (car_len, car_wid) in pixels so that the car,
    after being rotated to parking_angle_deg, fills about fill_factor of
    the slot rectangle (slot_w × slot_h) while preserving the original
    image aspect ratio.

    A per-vehicle CAR_FILL_MULTIPLIERS value is applied on top of fill_factor
    so each car type can be individually scaled up or down.

    car_len = width of the unrotated car (horizontal extent)
    car_wid = height of the unrotated car (vertical extent)
    """
    base_angle = CAR_IMAGE_BASE_ANGLES.get(car_type, 0)

    # Apply per-vehicle size multiplier
    effective_fill = fill_factor * CAR_FILL_MULTIPLIERS.get(car_type, 1.0)

    if vehicle_rgba is not None:
        img_h, img_w = vehicle_rgba.shape[:2]
        aspect = img_w / img_h
    else:
        aspect = 90.0 / 44.0            # fallback drawn sprite ratio

    # Effective rotation angle relative to the image's native orientation
    rotation = parking_angle_deg - base_angle
    rad = math.radians(rotation)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))

    # After rotating an (aspect*h, h) rectangle by `rotation` degrees, the
    # axis-aligned bounding box becomes:
    #   bb_w = h * (aspect * cos_a + sin_a)
    #   bb_h = h * (aspect * sin_a + cos_a)
    # We want bb_w <= slot_w * fill  AND  bb_h <= slot_h * fill.

    denom_w = aspect * cos_a + sin_a
    denom_h = aspect * sin_a + cos_a

    # Avoid division by a near-zero denominator
    denom_w = max(denom_w, 0.01)
    denom_h = max(denom_h, 0.01)

    h_from_w = (slot_w * effective_fill) / denom_w
    h_from_h = (slot_h * effective_fill) / denom_h

    car_h = min(h_from_w, h_from_h)
    car_wid = max(15, int(round(car_h)))
    car_len = max(20, int(round(car_h * aspect)))

    return car_len, car_wid


def draw_car(canvas, cx, cy, angle_deg, car_type, vehicle_rgba,
             car_len=90, car_wid=44):
    """
    If vehicle_rgba is available:
        1. Resize to (car_len × car_wid)  — the UNROTATED car size.
        2. Rotate to the desired angle.
        3. Alpha-overlay at the rotated sprite's own size (no further resize).
    Otherwise: draw the fallback vector sprite.
    """
    base_angle = CAR_IMAGE_BASE_ANGLES.get(car_type, 0)

    if vehicle_rgba is not None:
        # Step 1 — resize the original (unrotated) image to car dimensions
        resized = cv2.resize(vehicle_rgba, (car_len, car_wid),
                             interpolation=cv2.INTER_AREA)

        # Step 2 — rotate to match the desired heading
        rotation_needed = angle_deg - base_angle
        rotated = rotate_image(resized, rotation_needed)

        # Step 3 — overlay at the rotated image's native size
        overlay_rgba(canvas, rotated, cx, cy)
    else:
        draw_fallback_car(canvas, cx, cy, angle_deg, car_type,
                          length=car_len, width=car_wid)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PARKING LOGIC (identical to your original)
# ══════════════════════════════════════════════════════════════════════════════

def load_assets():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    img   = cv2.imread(IMAGE_PATH)
    if img is None:
        sys.exit("Could not load Reference.jpg — check IMAGE_PATH.")
    with open(SLOTS_PKL_PATH, "rb") as f:
        slots = pickle.load(f)
    return model, img, slots


def detect_slots(model, img, slots):
    slots_info = []
    for i, (x, y, w, h) in enumerate(slots):
        x = max(0, x); y = max(0, y)
        x2 = min(img.shape[1], x+w); y2 = min(img.shape[0], y+h)
        crop = img[y:y2, x:x2]
        if crop.size == 0:
            slots_info.append({"index":i,"x":x,"y":y,"w":w,"h":h,"status":"Occupied"})
            continue
        inp = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp,(64,64)).astype(np.float32)/255.0
        inp = np.expand_dims(inp,0)
        pred = model.predict(inp,verbose=0)[0][0]
        slots_info.append({"index":i,"x":x,"y":y,"w":w,"h":h,
                           "status":"Empty" if pred<0.5 else "Occupied"})
    return slots_info


def _neighbor_penalty(slots_info, i):
    left  = i>0                  and slots_info[i-1]["status"]=="Empty"
    right = i<len(slots_info)-1  and slots_info[i+1]["status"]=="Empty"
    if left and right: return 0
    if left or right:  return 50
    return 200


def find_best_slot(slots_info, car_type, img_shape):
    exit_pt  = (img_shape[1], img_shape[0])
    min_area = CAR_SIZES.get(car_type, 1500)
    best_slot, best_score = None, float("inf")
    for i,slot in enumerate(slots_info):
        if slot["status"]!="Empty": continue
        area=slot["w"]*slot["h"]
        if area<min_area: continue
        cx=slot["x"]+slot["w"]//2; cy=slot["y"]+slot["h"]//2
        ed=math.hypot(cx-ENTRY_POINT[0],cy-ENTRY_POINT[1])
        xd=math.hypot(cx-exit_pt[0],cy-exit_pt[1])
        penalty=_neighbor_penalty(slots_info,i) if car_type=="suv" else 0
        score=ed+xd*0.5+(1000/area)+penalty*3
        if score<best_score: best_score,best_slot=score,slot
    return best_slot,None


# ══════════════════════════════════════════════════════════════════════════════
#  BASE FRAME + PATH + ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

# Height of the extra road lane appended below the parking image (pixels).
# Sized so the deepest point of a large vehicle traveling downward (nose at
# EXTRA_ROAD_Y + car_len/2) never overflows the canvas bottom edge.
#   Original  130 px  →  ×1.35  →  176 px  →  ×1.35  →  238 px  (total ×1.82)
EXTRA_ROAD_H = 238


def _draw_extra_road(canvas, img_orig_h):
    """
    Paint a realistic extra road strip on `canvas` starting at y = img_orig_h.
    The strip is EXTRA_ROAD_H px tall and spans the full image width.
    Matches the gray asphalt + white dash + directional arrows of Reference.jpg.
    """
    W = canvas.shape[1]
    y0 = img_orig_h          # top of the extra road
    y1 = y0 + EXTRA_ROAD_H   # bottom of the strip

    # ── Asphalt base (lighter than the original road) ───────────────────────
    cv2.rectangle(canvas, (0, y0), (W, y1), (110, 110, 110), -1)

    # ── Subtle asphalt texture (noise) ──────────────────────────────────────
    noise = np.random.randint(0, 18, (EXTRA_ROAD_H, W), dtype=np.uint8)
    road_roi = canvas[y0:y1]
    road_roi[:, :, 0] = np.clip(road_roi[:, :, 0].astype(int) + noise - 9, 95, 130).astype(np.uint8)
    road_roi[:, :, 1] = road_roi[:, :, 0]
    road_roi[:, :, 2] = road_roi[:, :, 0]

    # ── Road edges (dark kerb lines) ─────────────────────────────────────────
    cv2.line(canvas, (0, y0),     (W, y0),     (50, 50, 50), 3)
    cv2.line(canvas, (0, y1 - 1), (W, y1 - 1), (50, 50, 50), 3)

    # ── White dashed centre line at mid-height of the strip ──────────────────
    cy = y0 + EXTRA_ROAD_H // 2
    dash_len = 55
    gap_len  = 35
    x = 0
    while x < W:
        x2 = min(x + dash_len, W)
        cv2.line(canvas, (x, cy), (x2, cy), (230, 230, 230), 2)
        x += dash_len + gap_len

    # ── Directional arrows (←) pointing LEFT — car drives right→down→left ───
    arrow_y = y0 + EXTRA_ROAD_H * 3 // 4   # lower half of the lane
    # Spread arrows evenly; tip of arrow faces left (←)
    arrow_xs = [1760, 1440, 1120, 800, 480, 160]
    for ax in arrow_xs:
        tip_x = ax          # tip of the arrowhead (leftmost point)
        tail_x = ax + 50    # tail of the shaft
        if tail_x > W or tip_x < 0:
            continue
        # shaft
        cv2.line(canvas, (tail_x, arrow_y), (tip_x, arrow_y), (210, 210, 210), 2)
        # arrowhead pointing left
        pts = np.array([
            [tip_x,      arrow_y],
            [tip_x + 14, arrow_y - 9],
            [tip_x + 14, arrow_y + 9],
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (210, 210, 210))


# Teal best-slot highlight colour  (BGR order for OpenCV)
_BEST_TEAL      = (0xD1, 0xFF, 0x00)    # #00FFD1 in BGR  → (209, 255, 0)
_BEST_TEAL_BGR  = (209, 255, 0)          # same, explicit


def _draw_best_slot_glow(frame, x, y, w, h):
    """
    Draw a vivid teal (#00FFD1) highlight with a multi-ring outer glow around
    the best parking slot.  Layers fade outward; a bold border + inner ring
    frame the slot; a ★ BEST label is stamped inside for unmistakeable clarity.
    """
    TEAL       = (209, 255,   0)   # #00FFD1 in BGR  (bright teal)
    TEAL_MID   = (180, 240,   0)   # mid-brightness ring
    TEAL_DIM   = (130, 195,   0)   # outermost soft halo
    WHITE_TEAL = (230, 255, 200)   # near-white for inner contrast

    # ── 5-ring outer glow: each ring expands further, alpha decreasing ────────
    glow_layers = [
        (22, TEAL_DIM, 0.10),   # outermost halo
        (16, TEAL_DIM, 0.17),
        (11, TEAL_MID, 0.26),
        ( 7, TEAL_MID, 0.35),
        ( 4, TEAL,     0.45),   # closest ring — bright and sharp
    ]
    for expand, colour, alpha in glow_layers:
        gx1 = max(0, x - expand)
        gy1 = max(0, y - expand)
        gx2 = min(frame.shape[1] - 1, x + w + expand)
        gy2 = min(frame.shape[0] - 1, y + h + expand)
        tmp = frame.copy()
        cv2.rectangle(tmp, (gx1, gy1), (gx2, gy2), colour, -1)
        cv2.addWeighted(tmp, alpha, frame, 1.0 - alpha, 0, frame)

    # ── Interior teal fill (55 % opacity — vivid but lets the background show) ─
    fill_tmp = frame.copy()
    cv2.rectangle(fill_tmp, (x, y), (x + w, y + h), TEAL, -1)
    cv2.addWeighted(fill_tmp, 0.55, frame, 0.45, 0, frame)

    # ── Thick outer border (5 px) ───────────────────────────────────────────────
    cv2.rectangle(frame, (x - 1, y - 1), (x + w + 1, y + h + 1), WHITE_TEAL, 1)
    cv2.rectangle(frame, (x,     y    ), (x + w,     y + h    ), TEAL,       5)

    # ── Inner double ring ───────────────────────────────────────────────────────
    cv2.rectangle(frame, (x + 5, y + 5), (x + w - 5, y + h - 5), TEAL_MID, 2)
    cv2.rectangle(frame, (x + 8, y + 8), (x + w - 8, y + h - 8), WHITE_TEAL, 1)



def build_base_frame(img, slots_info, best_slot, best_pair):
    # ── Append the extra road below the original image ───────────────────────
    orig_h, W = img.shape[:2]
    extra = np.zeros((EXTRA_ROAD_H, W, 3), dtype=np.uint8)
    frame = np.vstack([img.copy(), extra])

    overlay = frame.copy()
    for i, slot in enumerate(slots_info):
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        is_best = ((best_pair is None and best_slot and slot["index"] == best_slot["index"]) or
                   (best_pair is not None and i in (best_pair, best_pair + 1)))
        if is_best:
            continue   # best slot drawn separately with glow — skip here
        color = (109, 238, 109) if slot["status"] == "Empty" else (30, 30, 200)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    for i, slot in enumerate(slots_info):
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        is_best = ((best_pair is None and best_slot and slot["index"] == best_slot["index"]) or
                   (best_pair is not None and i in (best_pair, best_pair + 1)))
        if is_best:
            # Draw the teal glow for the best slot
            _draw_best_slot_glow(frame, x, y, w, h)
            # Label in matching teal
            cv2.putText(frame, str(i + 1), (x + 3, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (209, 255, 0), 1)
        else:
            color = (109, 238, 109) if slot["status"] == "Empty" else (30, 30, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, str(i + 1), (x + 3, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # ── Draw the extra road on the padded section ────────────────────────────
    _draw_extra_road(frame, orig_h)

    # ── Legend bar at the very bottom ───────────────────────────────────────
    hi = frame.shape[0]
    cv2.rectangle(frame, (0, hi - 22), (frame.shape[1], hi), (15, 15, 15), -1)
    cv2.putText(frame, "TEAL=Best  GREEN=Empty  RED=Occupied  |  Q=quit",
                (10, hi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)
    return frame


def build_path(img_shape, slot_cx, slot_cy, slot_x, slot_y, slot_w, slot_h, car_type):
    """
    Build a list of (x, y, heading°) waypoints from the road entry to the
    target slot, following the ACTUAL road layout of Reference.jpg.

    Road layout (1920 × 1080):
    ┌──────── TOP ROW (Slots 0-13, y < 275) ──────────────────┐
    │  entrance faces road below                               │
    ├──────────────────────────────────────────────┬────────────┤
    │  ← ← ←  UPPER LANE (left-bound)  ← ← ← ← │            │
    │  - - - - - - - - - - - - - - - - - - - - -  │  RIGHT     │
    │  → → →  LOWER LANE (y≈490, right-bound)  →  │  COLUMN    │
    │  ENTRY ⇒  car enters here from left edge    │  (14-20)   │
    ├──────────────────────────┐ open area (road)  │            │
    │  MIDDLE ROW (21-30)      │  x ≈ 1200-1620   │            │
    │  (y 590-840, face road)  │                   │            │
    ├──────────────────────────┤  car can drive     │            │
    │  BOTTOM ROW (31-40)      │  down here for     │            │
    │  (y 843-1040, face down) │  bottom row        │            │
    └──────────────────────────┴───────────────────┴────────────┘

    Traffic flow (counter-clockwise):
      → enter left on lower lane  →  turn down at right side  →
      ↓ drive down past row ends  →  turn left along bottom   →
      ← drive left to slot column →  turn up into bottom slot ↑
    """
    H, W = img_shape[:2]

    # ── Road geometry (measured from Reference.jpg) ──────────────────────
    LANE_Y        = 490     # Centre of the lower (right-bound) driving lane
    ROAD_TOP      = 275     # y where the road meets the top-row slot bottoms
    RIGHT_TURN_X  = 1400    # x where a car turns down (open area past rows)
    TURN_R        = 80      # Turning-radius offset for smooth curves

    # Extra road strip centre-line y (appended below the original image)
    # Original image height ≈ 1080;  extra road is EXTRA_ROAD_H px tall.
    EXTRA_ROAD_Y  = H + EXTRA_ROAD_H // 2   # mid-height of the new road lane

    # ── Classify the target slot ─────────────────────────────────────────
    is_top    = slot_cy < ROAD_TOP                  # Slots 0-13
    is_right  = slot_x >= 1620                      # Slots 14-20 (horizontal)
    is_bottom = slot_cy > 843 and slot_x < 1620     # Slots 31-40
    # Everything else = middle row (slots 21-30)
    # NOTE: hatchback uses the same path logic as sedan/suv (no special override)

    if is_top:
        # ── TOP ROW ─────────────────────────────────────────────────────
        # Drive right along the lower lane → curve upward → park (270°).
        approach_x = max(20, slot_cx - TURN_R)
        return [
            (0,          LANE_Y, 0),              # enter from left edge
            (approach_x, LANE_Y, 0),              # drive right to pre-turn
            (slot_cx,    LANE_Y - TURN_R, 270),   # curve upward
            (slot_cx,    slot_cy, 270),            # drive up into the slot
        ]

    elif is_right:
        # ── RIGHT COLUMN (horizontal slots, entrance on the left) ───────
        # Drive right along the lane → adjust y → drive right into slot.
        approach_x = slot_x - 20
        vert_angle = 270 if slot_cy < LANE_Y else 90
        return [
            (0,          LANE_Y,  0),             # enter from left
            (approach_x, LANE_Y,  0),             # drive right to column
            (approach_x, slot_cy, vert_angle),    # adjust height to slot row
            (slot_cx,    slot_cy, 0),             # drive right into the slot
        ]

    elif is_bottom:
        # ── BOTTOM ROW — arc waypoints at every corner so heading only
        #    rotates during the short TURN_R-pixel arc; the long straights
        #    keep a constant heading and the car travels perfectly straight.
        #
        # Route:
        #   ① Enter from left on main LANE_Y
        #   ② Drive right (heading 0°) to the turn column
        #   ③ Short arc: turn from 0°→90° while descending TURN_R px
        #   ④ Drive straight down (heading 90°) to the extra road
        #   ⑤ Short arc: turn from 90°→180° while moving left TURN_R px
        #   ⑥ Drive straight left (heading 180°) to the slot column
        #   ⑦ Short arc: turn from 180°→270° while ascending TURN_R px
        #   ⑧ Drive straight up (heading 270°) into the slot
        return [
            (0,                          LANE_Y,                  0),   # ① enter
            (RIGHT_TURN_X,               LANE_Y,                  0),   # ② reach corner
            (RIGHT_TURN_X,               LANE_Y + TURN_R,         90),  # ③ arc turn ↓
            (RIGHT_TURN_X,               EXTRA_ROAD_Y,            90),  # ④ straight down
            (RIGHT_TURN_X - TURN_R,      EXTRA_ROAD_Y,            180), # ⑤ arc turn ←
            (slot_cx,                    EXTRA_ROAD_Y,            180), # ⑥ straight left
            (slot_cx,                    EXTRA_ROAD_Y - TURN_R,   270), # ⑦ arc turn ↑
            (slot_cx,                    slot_cy,                 270), # ⑧ straight up
        ]

    else:
        # ── MIDDLE ROW ──────────────────────────────────────────────────
        # Drive right along the lower lane → curve downward → park (90°).
        approach_x = max(20, slot_cx - TURN_R)
        return [
            (0,          LANE_Y, 0),              # enter from left edge
            (approach_x, LANE_Y, 0),              # drive right to pre-turn
            (slot_cx,    LANE_Y + TURN_R, 90),    # curve downward
            (slot_cx,    slot_cy, 90),             # drive down into the slot
        ]


# ── Road geometry constants (exported for use by app.py) ─────────────────────
ROAD_LANE_Y       = 490
ROAD_TOP          = 275
ROAD_RIGHT_TURN_X = 1400


def build_exit_path(img_shape, slot_cx, slot_cy, slot_x, slot_y, slot_w, slot_h, car_type):
    """
    Build the exit path for a parked car by reversing its entry waypoints.
    The heading of each reversed waypoint is flipped by 180° so the car
    faces the direction it is actually travelling (backwards out of the slot).
    The final waypoint drives the car off the left edge of the image.
    """
    entry = build_path(img_shape, slot_cx, slot_cy,
                       slot_x, slot_y, slot_w, slot_h, car_type)

    H, W = img_shape[:2]
    LANE_Y = 490
    EXTRA_ROAD_Y = H + EXTRA_ROAD_H // 2

    # Reverse the waypoints (car retraces its entry route backwards)
    reversed_wps = list(reversed(entry))

    # Flip every heading by 180° so the car faces the direction of travel
    # (when reversing, the nose points opposite to the original entry heading)
    exit_wps = [(x, y, (a + 180) % 360) for x, y, a in reversed_wps]

    # Ensure the final leg merges to the main lane before exiting
    # Last reversed point is entry (0, LANE_Y, 180°) — car faces LEFT.
    # Append exit off the LEFT edge.
    last_x, last_y, last_a = exit_wps[-1]
    if last_x != 0 or last_y != LANE_Y:
        # Make sure we merge onto the lane before going off-screen
        exit_wps.append((0, LANE_Y, 180))

    # Drive off the left edge
    exit_wps.append((-150, LANE_Y, 180))

    return exit_wps


def smoothstep(t):
    return t*t*(3-2*t)


def interpolate_path(waypoints, steps=90):
    """
    Interpolate waypoints into per-frame (x, y, heading) tuples.

    Each segment gets  max(MIN_STEPS, int(dist / PIXELS_PER_STEP))  frames:
      • Short segments (turns, ~80 px) → floored at MIN_STEPS so they
        always animate smoothly — no sharp snapping.
      • Long segments (straight runs, 1000-1500 px on the bottom row)
        → get proportionally more frames so the car doesn't rocket across.
    """
    MIN_STEPS      = 90    # minimum frames per segment (keeps turns smooth)
    PIXELS_PER_STEP = 8    # pixels per frame on straight runs  (lower = slower)

    frames = []
    for i in range(len(waypoints) - 1):
        x0, y0, a0 = waypoints[i]
        x1, y1, a1 = waypoints[i + 1]

        dist = math.hypot(x1 - x0, y1 - y0)
        n    = max(MIN_STEPS, int(dist / PIXELS_PER_STEP))

        da = ((a1 - a0 + 180) % 360) - 180     # shortest angular path
        for s in range(n + 1):
            t = smoothstep(s / n)
            frames.append((
                int(x0 + (x1 - x0) * t),
                int(y0 + (y1 - y0) * t),
                a0 + da * t,
            ))
    return frames





def animate(base_frame, waypoints, slot_w, slot_h, car_type, vehicle_rgba):
    WIN = "Smart Parking"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(1400,base_frame.shape[1]), min(900,base_frame.shape[0]))
    FPS=30; delay=1000//FPS

    # ── Compute proper car size ───────────────────────────────────────────
    # Use the FINAL waypoint angle (the parked orientation) to calculate
    # how large the car should be so it fills the slot nicely.
    final_angle = waypoints[-1][2]
    car_len, car_wid = compute_car_dimensions(vehicle_rgba, slot_w, slot_h,
                                              final_angle, car_type,
                                              fill_factor=1.32)
    print(f"  [Car sprite size: {car_len}x{car_wid} px  |  "
          f"Slot: {slot_w}x{slot_h} px  |  Park angle: {final_angle:.0f} deg]")

    frames = interpolate_path(waypoints, steps=50)
    trail  = []

    for fi,(fx,fy,fa) in enumerate(frames):
        frame = base_frame.copy()
        trail.append((fx,fy))
        if len(trail)>1:
            for ti in range(1,len(trail)):
                if ti%8<4: cv2.line(frame,trail[ti-1],trail[ti],(0,190,210),1)
        draw_car(frame, fx, fy, fa, car_type, vehicle_rgba, car_len, car_wid)
        pct = int(fi/max(len(frames)-1,1)*100)
        cv2.rectangle(frame,(8,6),(310,48),(15,15,15),-1)
        cv2.putText(frame,f"Navigating...  {pct}%",(16,34),
                    cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,215,255),2)
        cv2.imshow(WIN,frame)
        if cv2.waitKey(delay)&0xFF==ord('q'):
            cv2.destroyAllWindows(); return

    fx,fy,fa = frames[-1]
    for step in range(25):
        a = step/24.0
        frame = base_frame.copy()
        car_layer = frame.copy()
        draw_car(car_layer, fx, fy, fa, car_type, vehicle_rgba, car_len, car_wid)
        cv2.addWeighted(car_layer,a,frame,1-a,0,frame)
        cv2.rectangle(frame,(8,6),(230,48),(15,15,15),-1)
        cv2.putText(frame,"Parking...",(16,34),cv2.FONT_HERSHEY_SIMPLEX,0.72,(50,255,120),2)
        cv2.imshow(WIN,frame)
        if cv2.waitKey(delay)&0xFF==ord('q'):
            cv2.destroyAllWindows(); return

    frame = base_frame.copy()
    draw_car(frame, fx, fy, fa, car_type, vehicle_rgba, car_len, car_wid)
    cv2.rectangle(frame,(8,6),(570,48),(15,15,15),-1)
    cv2.putText(frame,f"{car_type.upper()} parked!  Press any key to exit.",
                (16,34),cv2.FONT_HERSHEY_SIMPLEX,0.72,(50,255,120),2)
    cv2.imshow(WIN,frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(VEHICLES_DIR, exist_ok=True)

    print("Loading model and assets...")
    model, img, slots = load_assets()

    car_type = input("Enter vehicle type (hatchback / sedan / suv): ").lower().strip()
    if car_type not in CAR_SIZES:
        print("Invalid — defaulting to sedan")
        car_type = "sedan"

    print(f"Looking for vehicle image in:  {VEHICLES_DIR}")
    vehicle_rgba = load_vehicle_image(car_type)

    print("Running slot detection...")
    slots_info = detect_slots(model, img, slots)
    avail = sum(1 for s in slots_info if s["status"]=="Empty")
    print(f"  {avail} empty / {len(slots_info)} total slots")

    best_slot,_ = find_best_slot(slots_info,car_type,img.shape)
    if best_slot is None:
        print(f"No suitable slot found for {car_type.upper()}")
        return

    slot_x=best_slot["x"]; slot_y=best_slot["y"]
    slot_w=best_slot["w"]; slot_h=best_slot["h"]
    print(f"Best slot for {car_type.upper()}: Slot {best_slot['index']+1}")

    slot_cx = slot_x+slot_w//2
    slot_cy = slot_y+slot_h//2

    waypoints = build_path(img.shape,slot_cx,slot_cy,slot_x,slot_y,slot_w,slot_h,car_type)
    base = build_base_frame(img,slots_info,best_slot,None)

    print("Launching animation...  (Q to quit early)")
    animate(base,waypoints,slot_w,slot_h,car_type,vehicle_rgba)


if __name__ == "__main__":
    main()