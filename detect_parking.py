import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("parking_model.h5")

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(r"C:\Users\DELL\OneDrive\Documents\java final prep\Parking_Slot_Project\frames\frame_031.jpg")

if img is None:
    print("❌ Error loading image")
    exit()

# -----------------------------
# LOAD SLOT COORDINATES
# -----------------------------
with open(r"C:\Users\DELL\OneDrive\Desktop\parking lot\slots.pkl", "rb") as f:
    slots = pickle.load(f)

# -----------------------------
# IMPORTANT: CHECK CLASS ORDER
# -----------------------------
# ⚠️ UPDATE THIS BASED ON YOUR TRAINING OUTPUT
# Run this once in training:
# print(train_ds.class_names)

# Example (CHANGE if needed):
CLASS_NAMES = ["Empty", "Occupied"]
# If your output was ['Occupied', 'Empty'], then swap above

# -----------------------------
# PROCESS SLOTS
# -----------------------------
available_slots = 0

for i, (x, y, w, h) in enumerate(slots):

    # Crop slot
    slot_img = img[y:y+h, x:x+w]

    # -----------------------------
    # PREPROCESSING (MATCH TRAINING)
    # -----------------------------
    
    # 🔥 FIX 1: Convert BGR → RGB
    slot_img = cv2.cvtColor(slot_img, cv2.COLOR_BGR2RGB)

    # Resize
    slot_img = cv2.resize(slot_img, (64, 64))

    # Normalize
    slot_img = slot_img / 255.0

    # Expand dims
    slot_img = np.expand_dims(slot_img, axis=0)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    prediction = model.predict(slot_img, verbose=0)[0][0]

    # Debug print
    print(f"Slot {i+1} prediction value: {prediction:.4f}")

    # -----------------------------
    # CLASS DECISION
    # -----------------------------
    if prediction < 0.5:
        label = CLASS_NAMES[0]  # Empty
        color = (0, 255, 0)     # Green
        available_slots += 1
    else:
        label = CLASS_NAMES[1]  # Occupied
        color = (0, 0, 255)     # Red

    # -----------------------------
    # DRAW BOX
    # -----------------------------
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Slot number + status
    cv2.putText(img,
                f"{i+1}:{label}",
                (x+5, y+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2)

# -----------------------------
# FINAL COUNT DISPLAY
# -----------------------------
total_slots = len(slots)
occupied_slots = total_slots - available_slots

print("\n✅ Available Slots:", available_slots)
print("🚗 Occupied Slots:", occupied_slots)

# Display on image
cv2.putText(img,
            f"Available: {available_slots}/{total_slots}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3)

# -----------------------------
# SHOW RESULT
# -----------------------------
cv2.imshow("Parking Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()