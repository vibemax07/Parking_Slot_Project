import cv2
import os

video_path = r"C:\Users\91630\Videos\Captures\(40) Car parking lot viewed from above timelapse, Aerial view - YouTube and 2 more pages - Personal - Microsoft​ Edge 2026-01-29 13-34-46.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
print("Video opened:", cap.isOpened())

frame_count = 0
saved_count = 0
SKIP_FRAMES = 7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % SKIP_FRAMES == 0:
        filename = f"frame_{saved_count:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames.")
print("Current working directory:", os.getcwd())
