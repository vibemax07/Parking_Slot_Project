import cv2
import numpy as np
import pickle
import math
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model(r"C:\Parking Slot\Parking_Slot_Project\parking_model.h5")

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(r"C:\Parking Slot\Parking_Slot_Project\Reference.jpg")

if img is None:
    print("❌ Error loading image")
    exit()

# -----------------------------
# LOAD SLOT COORDINATES
# -----------------------------
with open(r"C:\Parking Slot\Parking_Slot_Project\slots.pkl", "rb") as f:
    slots = pickle.load(f)

# -----------------------------
# VEHICLE INPUT
# -----------------------------
car_type = input("Enter vehicle type (hatchback / sedan / suv / truck): ").lower().strip()

CAR_SIZES = {
    "hatchback": 1500,
    "sedan":     2500,
    "suv":       3500,
    "truck":     3500   # each individual slot must still be big enough
}

if car_type not in CAR_SIZES:
    print("⚠ Invalid input! Defaulting to hatchback")
    car_type = "hatchback"

min_area = CAR_SIZES[car_type]

# -----------------------------
# ENTRY AND EXIT POINTS
# Define these based on your actual parking lot layout
# Entry = where cars come in, Exit = where cars leave
# -----------------------------
ENTRY_POINT = (0, 0)                              # top-left corner
EXIT_POINT  = (img.shape[1], img.shape[0])        # bottom-right corner

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# -----------------------------
# NEIGHBOR PENALTY (SUV only)
# -----------------------------
def get_neighbor_penalty(slots_info, i):
    left_empty  = i > 0 and slots_info[i-1]["status"] == "Empty"
    right_empty = i < len(slots_info) - 1 and slots_info[i+1]["status"] == "Empty"

    if left_empty and right_empty:
        return 0    # both sides open — easy to maneuver
    elif left_empty or right_empty:
        return 50   # one side open — manageable
    else:
        return 200  # both sides occupied — tight squeeze

# -----------------------------
# PROCESS SLOTS (DETECTION)
# -----------------------------
available_slots = 0
slots_info = []

for i, (x, y, w, h) in enumerate(slots):

    # Clamp to image bounds
    x  = max(0, x)
    y  = max(0, y)
    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)

    slot_img = img[y:y2, x:x2]

    if slot_img.size == 0:
        slots_info.append({
            "index": i, "x": x, "y": y, "w": w, "h": h, "status": "Occupied"
        })
        continue

    # BGR → RGB to match training pipeline
    slot_img = cv2.cvtColor(slot_img, cv2.COLOR_BGR2RGB)
    slot_img = cv2.resize(slot_img, (64, 64))
    slot_img = slot_img.astype(np.float32) / 255.0
    slot_img = np.expand_dims(slot_img, axis=0)

    prediction = model.predict(slot_img, verbose=0)[0][0]

    if prediction < 0.5:
        status = "Empty"
        color  = (0, 255, 0)   # Green = Empty
        available_slots += 1
    else:
        status = "Occupied"
        color  = (0, 0, 255)   # Red = Occupied

    slots_info.append({
        "index": i,
        "x": x, "y": y,
        "w": w, "h": h,
        "status": status
    })

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f"{i+1}", (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# -----------------------------
# BEST SLOT LOGIC
# -----------------------------
best_slot       = None
best_score      = float('inf')
best_pair_index = None   # used for truck (stores index of first slot in pair)

# --- TRUCK: find best pair of adjacent empty slots ---
if car_type == "truck":
    for i in range(len(slots_info) - 1):
        s1 = slots_info[i]
        s2 = slots_info[i + 1]

        if s1["status"] != "Empty" or s2["status"] != "Empty":
            continue

        # Both slots must individually meet the size requirement
        if (s1["w"] * s1["h"]) < min_area or (s2["w"] * s2["h"]) < min_area:
            continue

        # Use midpoint of the pair as reference
        mid_x = (s1["x"] + s2["x"] + s2["w"]) // 2
        mid_y = (s1["y"] + s1["h"]) // 2

        entry_dist = calculate_distance(mid_x, mid_y, ENTRY_POINT[0], ENTRY_POINT[1])
        exit_dist  = calculate_distance(mid_x, mid_y, EXIT_POINT[0],  EXIT_POINT[1])

        score = entry_dist + (exit_dist * 0.5)   # exit matters, but less than entry

        print(f"Truck pair ({i+1},{i+2}) → Entry:{entry_dist:.1f}, Exit:{exit_dist:.1f}, Score:{score:.2f}")

        if score < best_score:
            best_score      = score
            best_slot       = s1       # highlight starts from first slot
            best_pair_index = i

# --- ALL OTHER VEHICLES: score each empty slot ---
else:
    for i, slot in enumerate(slots_info):

        if slot["status"] != "Empty":
            continue

        area = slot["w"] * slot["h"]

        # Skip if slot is too small for this vehicle
        if area < min_area:
            continue

        cx = slot["x"] + slot["w"] // 2
        cy = slot["y"] + slot["h"] // 2

        entry_dist = calculate_distance(cx, cy, ENTRY_POINT[0], ENTRY_POINT[1])
        exit_dist  = calculate_distance(cx, cy, EXIT_POINT[0],  EXIT_POINT[1])

        # SUV gets neighbor penalty, others don't
        neighbor_penalty = get_neighbor_penalty(slots_info, i) 

        # Final score — lower is better
        # Entry distance weighted more than exit
        # Area term rewards larger slots
        score = (entry_dist * 1.0) + (exit_dist * 0.5) + (1000 / area) + (neighbor_penalty*3)

        print(f"Slot {i+1} → Entry:{entry_dist:.1f}, Exit:{exit_dist:.1f}, "
              f"Area:{area}, Penalty:{neighbor_penalty}, Score:{score:.2f}")

        if score < best_score:
            best_score = score
            best_slot  = slot

# -----------------------------
# HIGHLIGHT BEST SLOT
# -----------------------------
if best_slot is not None:

    if car_type == "truck" and best_pair_index is not None:
        # Highlight both slots in the pair
        s1 = slots_info[best_pair_index]
        s2 = slots_info[best_pair_index + 1]

        cv2.rectangle(img, (s1["x"], s1["y"]), (s1["x"]+s1["w"], s1["y"]+s1["h"]), (255, 0, 0), 4)
        cv2.rectangle(img, (s2["x"], s2["y"]), (s2["x"]+s2["w"], s2["y"]+s2["h"]), (255, 0, 0), 4)
        cv2.putText(img, "BEST", (s1["x"] + 5, s1["y"] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, "TRUCK", (s2["x"] + 5, s2["y"] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        mid_x = (s1["x"] + s2["x"] + s2["w"]) // 2
        mid_y = (s1["y"] + s1["h"]) // 2
        cv2.line(img, ENTRY_POINT, (mid_x, mid_y), (255, 255, 0), 2)

        print(f"\n✅ Best truck pair: Slot {s1['index']+1} + Slot {s2['index']+1}")

    else:
        x, y, w, h = best_slot["x"], best_slot["y"], best_slot["w"], best_slot["h"]

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
        cv2.putText(img, "BEST", (x + 5, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.line(img, ENTRY_POINT, (x + w // 2, y + h // 2), (255, 255, 0), 2)

        print(f"\n✅ Best slot for {car_type.upper()}: Slot {best_slot['index']+1}")

else:
    print(f"\n❌ No suitable slot found for {car_type.upper()}")

# -----------------------------
# FINAL DISPLAY
# -----------------------------
total_slots    = len(slots)
occupied_slots = total_slots - available_slots

print(f"\nAvailable : {available_slots}/{total_slots}")
print(f"Occupied  : {occupied_slots}/{total_slots}")

cv2.putText(img, f"Available: {available_slots}/{total_slots}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

cv2.imshow("Smart Parking System", img)
cv2.waitKey(0)
cv2.destroyAllWindows()