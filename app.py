import cv2
import mediapipe as mp
import numpy as np
import os
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Drawing for visualizing pose landmarks
mp_drawing = mp.solutions.drawing_utils

# Load Shirt Images from the specified directory
shirt_dir = 'Resources/Shirts'
shirt_images = [cv2.imread(os.path.join(shirt_dir, img), cv2.IMREAD_UNCHANGED) for img in os.listdir(shirt_dir)]
shirt_index = 0  # Initialize the index for selecting the current shirt

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Function to overlay the shirt on the user's body based on pose landmarks
def overlay_shirt(frame, shirt_img, landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    shoulder_width = int(np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - 
                                        np.array([right_shoulder.x, right_shoulder.y])) * frame.shape[1] * 1.7)
    shirt_height = int(shoulder_width * (shirt_img.shape[0] / shirt_img.shape[1]) * 1.28)

    x_center = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
    y_center = int((left_shoulder.y + left_hip.y) / 2 * frame.shape[0])

    angle = -np.degrees(np.arctan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x))

    try:
        shirt_img = cv2.resize(shirt_img, (shoulder_width, shirt_height))
    except Exception as e:
        print(f"Error resizing shirt: {e}")
        return frame

    M = cv2.getRotationMatrix2D((shoulder_width // 2, shirt_height // 2), angle, 1)
    shirt_img = cv2.warpAffine(shirt_img, M, (shoulder_width, shirt_height))

    x_start = x_center - shoulder_width // 2
    y_start = y_center - shirt_height // 2

    x_start = max(0, x_start)
    y_start = max(0, y_start)

    if y_start + shirt_height > frame.shape[0] or x_start + shoulder_width > frame.shape[1]:
        print("Shirt dimensions exceed frame boundaries.")
        return frame

    alpha = shirt_img[:, :, 3] / 255.0
    overlay = shirt_img[:, :, :3]

    for c in range(0, 3):
        frame[y_start:y_start+shirt_height, x_start:x_start+shoulder_width, c] = \
            (1 - alpha) * frame[y_start:y_start+shirt_height, x_start:x_start+shoulder_width, c] + \
            alpha * overlay[:, :, c]

    return frame

# Function to generate the video stream
def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                frame = overlay_shirt(frame, shirt_images[shirt_index], landmarks)
            except Exception as e:
                print(f"Error during overlay: {e}")

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert the frame to JPEG format for web streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')  # Create an HTML page to embed the video

# Run the app
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
