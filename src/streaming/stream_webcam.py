import cv2
import subprocess
import os, sys
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
WEBCAM_INDEX = 0
RTSP_URL = os.getenv('RTSP_URL')
FPS = 30

# Open the webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FFmpeg command
command = [
    'ffmpeg',
    '-y',                 # Overwrite output file if it exists
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',  # OpenCV uses BGR format
    '-s', f'{width}x{height}',
    '-r', str(FPS),
    '-i', '-',            # Input from stdin
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'rtsp',
    RTSP_URL
]

# Start the FFmpeg subprocess
p = subprocess.Popen(command, stdin=subprocess.PIPE)

print("Streaming webcam to RTSP server. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Write the frame to the stdin of the FFmpeg subprocess
        p.stdin.write(frame.tobytes())

except KeyboardInterrupt:
    print("Stopping stream...")
finally:
    p.stdin.close()
    p.wait()
    cap.release()
    print("Stream stopped.")