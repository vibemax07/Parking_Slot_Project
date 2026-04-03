# imports

import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# ── Load slot coordinates ──────────────────────────────────────────────────────
with open("slots.pkl", "rb") as f:
    parking_slots = pickle.load(f)

# ── Load model ─────────────────────────────────────────────────────────────────
model = load_model("parking_model.h5")

# ── Load reference frame ───────────────────────────────────────────────────────
frame = cv2.imread(r"C:\Users\DELL\OneDrive\Documents\java final prep\Parking_Slot_Project\Reference.jpg")

if frame is None:
    print("ERROR: Image not found. Check your path.")
    exit()

slot_status = []

# ── Check each slot ────────────────────────────────────────────────────────────
for i, slot in enumerate(parking_slots):
    x, y, w, h = slot

    # Clamp coordinates so crop never goes out of frame bounds
    x  = max(0, x)
    y  = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    slot_img = frame[y:y2, x:x2]

    # Skip corrupt/empty crops
    if slot_img.size == 0:
        slot_status.append("occupied")
        continue

    # FIX 1: Convert BGR → RGB to match training pipeline
    slot_img = cv2.cvtColor(slot_img, cv2.COLOR_BGR2RGB)

    slot_img = cv2.resize(slot_img, (64, 64))
    slot_img = slot_img.astype(np.float32) / 255.0
    slot_img = np.expand_dims(slot_img, axis=0)

    prediction = model.predict(slot_img, verbose=0)
    raw = prediction[0][0]

    # FIX 2: Corrected label logic (empty folder loads as 0 alphabetically)
    if raw > 0.5:
        slot_status.append("empty")
    else:
        slot_status.append("occupied")

# ── Find best (first) empty slot ───────────────────────────────────────────────
best_slot = next((i for i, s in enumerate(slot_status) if s == "empty"), None)

print(f"Best slot index : {best_slot}")
print(f"Slot statuses   : {slot_status}")

# ── Draw status rectangles ─────────────────────────────────────────────────────
for i, slot in enumerate(parking_slots):
    x, y, w, h = slot
    color = (0, 0, 255) if slot_status[i] == "empty" else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

# ── Highlight best slot in blue ────────────────────────────────────────────────
if best_slot is not None:
    x, y, w, h = parking_slots[best_slot]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

# ── Save and display ───────────────────────────────────────────────────────────
cv2.imwrite("result.jpg", frame)
print("Result saved as result.jpg")

cv2.imshow("Parking Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
