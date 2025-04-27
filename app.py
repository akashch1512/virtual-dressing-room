import cv2
import requests
import numpy as np
import os
from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO
import base64
import mediapipe as mp
from io import BytesIO
from PIL import Image
import time

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing for visualizing pose landmarks
mp_drawing = mp.solutions.drawing_utils

# Current selected shirt
current_shirt = None
generated_shirts = []  # Stores shirts generated from API

SERPAPI_KEY = "258ef6e2c5b5347e763500df9e2c9e1616bd04c7462a82ba0982ab099a883723" # api key from serpapi
REMOVE_BG_API_KEY = "6Pf6aFYYwLb9oRJ7XnztUVi1"  # Replace with your actual API key from removebg

# Capture video from webcam
cap = cv2.VideoCapture(0)


def remove_background(image_url):
    """Remove background using remove.bg API"""
    try:
        # First download the image
        response = requests.get(image_url)
        if response.status_code != 200:
            return None
            
        # Send to remove.bg API
        files = {'image_file': BytesIO(response.content)}
        data = {'size': 'auto'}
        headers = {'X-Api-Key': REMOVE_BG_API_KEY}
        
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files=files,
            data=data,
            headers=headers
        )
        
        if response.status_code == 200:
            # Convert to OpenCV format with alpha channel
            img = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        else:
            print(f"Remove.bg API error: {response.status_code} {response.text}")
            return None
            
    except Exception as e:
        print(f"Error in remove_background: {e}")
        return None

def overlay_shirt(frame, shirt_img, landmarks):
    if shirt_img is None:
        return frame

    try:
        # Get the relevant landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate shoulder width in pixels
        shoulder_width_px = int(np.linalg.norm(
            np.array([left_shoulder.x * frame.shape[1], left_shoulder.y * frame.shape[0]]) - 
            np.array([right_shoulder.x * frame.shape[1], right_shoulder.y * frame.shape[0]])
        ) * 1.5)  # Scale factor for better fit

        # Calculate shirt height maintaining aspect ratio
        shirt_aspect_ratio = shirt_img.shape[0] / shirt_img.shape[1]
        shirt_height_px = int(shoulder_width_px * shirt_aspect_ratio * 1.2)  # Scale factor for length
        
        # Resize shirt image
        shirt_img_resized = cv2.resize(shirt_img, (shoulder_width_px, shirt_height_px))
        
        # Calculate rotation angle based on shoulder slope
        angle = np.degrees(np.arctan2(
            left_shoulder.y * frame.shape[0] - right_shoulder.y * frame.shape[0],
            left_shoulder.x * frame.shape[1] - right_shoulder.x * frame.shape[1]
        ))
        
        # Rotate shirt image
        rotation_matrix = cv2.getRotationMatrix2D((shoulder_width_px // 2, shirt_height_px // 2), angle, 1)
        shirt_img_rotated = cv2.warpAffine(shirt_img_resized, rotation_matrix, (shoulder_width_px, shirt_height_px))
        
        # Calculate position to place the shirt (center between shoulders and hips)
        x_center = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
        y_center = int((left_shoulder.y + left_hip.y) / 2 * frame.shape[0])
        
        x_start = x_center - shoulder_width_px // 2
        y_start = y_center - shirt_height_px // 3
        
        # Ensure the shirt stays within frame boundaries
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(frame.shape[1], x_start + shoulder_width_px)
        y_end = min(frame.shape[0], y_start + shirt_height_px)
        
        # Adjust if shirt goes out of bounds
        if x_end <= x_start or y_end <= y_start:
            return frame
            
        shirt_img_rotated = shirt_img_rotated[:y_end-y_start, :x_end-x_start]
        
        # Extract the alpha channel and create mask
        if shirt_img_rotated.shape[2] == 4:
            alpha = shirt_img_rotated[:, :, 3] / 255.0
            overlay = shirt_img_rotated[:, :, :3]
        else:
            alpha = np.ones(shirt_img_rotated.shape[:2], dtype=np.float32)
            overlay = shirt_img_rotated
            
        # Overlay the shirt on the frame
        for c in range(0, 3):
            frame[y_start:y_end, x_start:x_end, c] = (
                (1 - alpha) * frame[y_start:y_end, x_start:x_end, c] + 
                alpha * overlay[:, :, c]
            )
            
    except Exception as e:
        print(f"Error in overlay_shirt: {e}")
        
    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            if current_shirt is not None:
                frame = overlay_shirt(frame, current_shirt, results.pose_landmarks.landmark)
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_shirts_from_prompt(prompt):
    global generated_shirts
    
    search_query = f"{prompt} men's shirt, high resolution, single shirt image only,only image must be there in a photo, no body parts or backgrounds, isolated shirt, high-definition, full size image, focused on fabric texture and details, no logos or distractions"
    params = {
        "engine": "google",
        "q": search_query,
        "tbm": "isch",
        "api_key": SERPAPI_KEY
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        
        generated_shirts = []
        for i, image in enumerate(data.get("images_results", [])[:4]):  # Limit to 4 images
            try:
                img_url = image["original"]
                print(f"Processing image {i+1}: {img_url}")
                
                # Remove background first
                shirt_img = remove_background(img_url)
                
                if shirt_img is not None:
                    generated_shirts.append({
                        'id': i,
                        'url': img_url,
                        'image': shirt_img
                    })
            except Exception as e:
                print(f"Error processing image {i}: {e}")
        
        return True, "Shirts generated successfully"
    except Exception as e:
        return False, f"API error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_shirts', methods=['POST'])
def generate_shirts():
    prompt = request.form.get('prompt', '').strip()
    if not prompt:
        return jsonify({'success': False, 'message': 'Please enter a prompt'})
    
    success, message = generate_shirts_from_prompt(prompt)
    if not success:
        return jsonify({'success': False, 'message': message})
    
    # Prepare shirt data for response
    shirt_data = []
    for shirt in generated_shirts:
        _, buffer = cv2.imencode('.png', shirt['image'])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        shirt_data.append({
            'id': shirt['id'],
            'url': shirt['url'],
            'image': img_base64
        })
    
    return jsonify({'success': True, 'shirts': shirt_data})

@app.route('/select_shirt', methods=['POST'])
def select_shirt():
    global current_shirt
    shirt_id = int(request.form.get('shirt_id'))
    
    if 0 <= shirt_id < len(generated_shirts):
        current_shirt = generated_shirts[shirt_id]['image']
        return jsonify({'success': True, 'message': f'Shirt {shirt_id} selected'})
    
    return jsonify({'success': False, 'message': 'Invalid shirt ID'})

if __name__ == '__main__':
    socketio.run(app, debug=True)