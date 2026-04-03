import cv2
import pickle

# Load image
img = cv2.imread(r"C:\\Users\DELL\\OneDrive\Documents\\java final prep\\Parking_Slot_Project\\Reference.jpg")

# Load slot coordinates
with open(r"C:\\Users\DELL\\OneDrive\Desktop\\parking lot\\slots.pkl", "rb") as f:
    slots = pickle.load(f)

# Draw each slot
for i, (x, y, w, h) in enumerate(slots):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"Slot {i}",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (254, 255, 254),
        1
    )
print(slots)

if img is None:
    print("good")

else:
    # Show image
    cv2.imshow("Parking Slots", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()