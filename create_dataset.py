import cv2
import os
import pickle

# Load slot coordinates
with open(r"C:\Users\DELL\OneDrive\Desktop\parking lot\slots.pkl", "rb") as f:
    slots = pickle.load(f)

frames_path = r"C:\Users\DELL\OneDrive\Documents\java final prep\Parking_Slot_Project\frames"   # folder where your 42 images are

image_count = 0

for frame_name in os.listdir(frames_path):

    frame_path = os.path.join(frames_path, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    for (x, y, w, h) in slots:

        slot_img = frame[y:y+h, x:x+w]
        slot_img = cv2.resize(slot_img, (64, 64))

        cv2.imshow("Slot", slot_img)
        key = cv2.waitKey(0)

        # Press:
        # E → empty
        # O → occupied
        # ESC → skip

        if key == ord('e'):
            save_path = f"dataset/empty/{image_count}.jpg"
            cv2.imwrite(save_path, slot_img)
            image_count += 1

        elif key == ord('o'):
            save_path = f"dataset/occupied/{image_count}.jpg"
            cv2.imwrite(save_path, slot_img)
            image_count += 1

cv2.destroyAllWindows()

print("✅ Dataset creation complete!")
