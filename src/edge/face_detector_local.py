import cv2
import time
import os, sys
from loguru import logger

from helper import non_max_suppression_fast

# --- Settings ---
VIDEO_SOURCE = 0  # webcam
OUTPUT_PATH = "src/streaming/faces_output.avi"
W, H = 640, 480

# Load Haar cascade from OpenCV's built-in path
haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(haar_path)

if face_cascade.empty():
    raise IOError("Failed to load Haar cascade XML.")

# Open video source
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

# Output video writer
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True) # create the output if it doesn't already exist
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"MJPG"),
    15.0,
    (W, H)
)

# FPS tracking
t0 = time.time()
frames = 0

logger.info("Starting Haar face detection...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (W, H))
    
    # Introduce gray - Make it faster as its gray and improve contrast and filter for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  
    gray = cv2.bilateralFilter(gray, 5, 75, 75)

    # Detect faces
    faces_start_time = time.time()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,   # smaller scaleFactor, better face detections but slower
        minNeighbors=3,    # more neighbours, lesser face detections
        minSize=(40, 40)  
    )
    faces = non_max_suppression_fast(faces, overlap_thresh=0.3)
    faces_end_time = time.time()
    faces_latency = (faces_end_time - faces_start_time) * 1000

    # Draw bounding boxes for faces
    for (x, y, w, h) in faces:
      pad = int(0.05 * w)
      cv2.rectangle(frame, (x + pad, y + pad), (x + w - pad, y + h - pad), (0, 255, 0), 2)

    # FPS counter
    frames += 1
    elapsed = time.time() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0

    # Overlay info
    cv2.putText(frame, f"Faces: {len(faces)} | FPS: {fps:.1f} | Latency: {faces_latency:.1f} ms",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 1)

    # Save & display
    out.write(frame)
    cv2.imshow("Haar Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
# cv2.destroyAllWindows()