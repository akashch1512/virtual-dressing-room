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

# Initialize MediaPipe Pose with more accurate parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Higher complexity for better accuracy
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize MediaPipe Drawing for visualizing pose landmarks
mp_drawing = mp.solutions.drawing_utils

# Current selected shirt
current_shirt = None
generated_shirts = []  # Stores shirts generated from API

SERPAPI_KEY = "258ef6e2c5b5347e763500df9e2c9e1616bd04c7462a82ba0982ab099a883723"

# extra api key with credts
# SERPAPI_KEY = "222e58038ed2c15178813840e2a4c10eed88d42b684bb9aae12731baa7b095f6"
REMOVE_BG_API_KEY = "BRvJ96jcvFbdtzd3MaVCzaRc"


# Capture video from webcam
cap = cv2.VideoCapture(0)

def auto_crop_image(img):
    """Auto-crop the image to remove transparent borders"""
    if img.shape[2] == 4:  # Check if image has alpha channel
        # Convert to grayscale alpha channel
        alpha = img[:,:,3]

        # Find non-zero coordinates
        coords = cv2.findNonZero(alpha)
        if coords is None:
            return img

        x, y, w, h = cv2.boundingRect(coords)

        # Add some padding (10% of the width/height)
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img.shape[1] - x, w + 2*pad_x)
        h = min(img.shape[0] - y, h + 2*pad_y)

        return img[y:y+h, x:x+w]
    return img

def remove_background(image_url):
    """Remove background using remove.bg API with auto-cropping and orientation handling"""
    try:
        # First download the image
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Error downloading image: {response.status_code}")
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
            # Convert to OpenCV format with alpha channel, handling orientation
            img = Image.open(BytesIO(response.content))
            # Fix orientation if needed
            if hasattr(img, '_getexif'):
                exif = img._getexif()
                if exif is not None and 274 in exif:  # 274 is the orientation tag
                    orientation = exif[274]
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

            # Auto-crop the image to remove empty space
            img_cropped = auto_crop_image(img_cv)

            return img_cropped
        else:
            print(f"Remove.bg API error: {response.status_code} {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Network error in remove_background: {e}")
        return None
    except Exception as e:
        print(f"Error in remove_background: {e}")
        return None

def calculate_shirt_position(landmarks, frame_shape):
    """Calculate optimal shirt position and size based on pose landmarks with improved logic"""
    try:
        # Get relevant landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate shoulder center and hip center
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2

        # Calculate torso length (vertical distance between shoulder and hip centers)
        torso_length_normalized = np.linalg.norm(
            np.array([shoulder_center_x, shoulder_center_y]) -
            np.array([hip_center_x, hip_center_y])
        )
        torso_length_px = int(torso_length_normalized * frame_shape[0] * 1.5)  # Adjusted scaling

        # Calculate shoulder width
        shoulder_width_px = int(np.linalg.norm(
            np.array([left_shoulder.x * frame_shape[1], left_shoulder.y * frame_shape[0]]) -
            np.array([right_shoulder.x * frame_shape[1], right_shoulder.y * frame_shape[0]])
        ) * 1.3)  # Adjusted scaling

        # Calculate shirt dimensions
        shirt_width = int(shoulder_width_px * 1.1)  # Slightly wider than shoulders
        shirt_height = int(torso_length_px * 1.2)  # Covers more of the torso

        # Calculate shirt center position (slightly below the shoulder center)
        center_x = int(shoulder_center_x * frame_shape[1])
        center_y = int((shoulder_center_y * 0.6 + hip_center_y * 0.4) * frame_shape[0])  # Adjusted vertical position

        # Calculate rotation angle based on shoulder orientation
        if right_shoulder.x > left_shoulder.x:
            angle = np.degrees(np.arctan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ))
        else:
            angle = np.degrees(np.arctan2(
                left_shoulder.y - right_shoulder.y,
                left_shoulder.x - right_shoulder.x
            ))

        return {
            'width': shirt_width,
            'height': shirt_height,
            'center_x': center_x,
            'center_y': center_y,
            'angle': angle
        }

    except Exception as e:
        print(f"Error in calculate_shirt_position: {e}")
        return None

def overlay_shirt(frame, shirt_img, landmarks):
    """Overlay the shirt onto the frame based on calculated position and orientation"""
    if shirt_img is None:
        return frame

    try:
        # Calculate shirt position and size
        position = calculate_shirt_position(landmarks, frame.shape)
        if position is None:
            return frame

        shirt_height, shirt_width, _ = shirt_img.shape

        # Calculate scaling factors based on desired width and height
        scale_width = position['width'] / shirt_width
        scale_height = position['height'] / shirt_height
        scale = max(scale_width, scale_height)  # Scale to fit within the calculated bounds

        # Resize shirt
        new_width = int(shirt_width * scale)
        new_height = int(shirt_height * scale)
        shirt_resized = cv2.resize(shirt_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Rotate shirt
        rotation_matrix = cv2.getRotationMatrix2D(
            (new_width // 2, new_height // 2),
            position['angle'],
            1
        )
        shirt_rotated = cv2.warpAffine(
            shirt_resized,
            rotation_matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_TRANSPARENT
        )

        # Calculate overlay position
        x_offset = position['center_x'] - new_width // 2
        y_offset = position['center_y'] - new_height // 3

        # Ensure no negative offsets
        y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + new_height)
        x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + new_width)

        # Extract the region of interest in the frame
        roi = frame[y1:y2, x1:x2]

        # Extract the corresponding region from the rotated shirt
        shirt_overlay = shirt_rotated[max(0, -y_offset):min(new_height, frame.shape[0] - y_offset),
                                     max(0, -x_offset):min(new_width, frame.shape[1] - x_offset)]

        # Create a mask for the shirt overlay
        if shirt_overlay.shape[2] == 4:
            mask = shirt_overlay[:, :, 3] / 255.0
            mask_inv = 1.0 - mask
        else:
            mask = np.ones(shirt_overlay.shape[:2], dtype=np.float32)
            mask_inv = np.zeros(shirt_overlay.shape[:2], dtype=np.float32)
            shirt_overlay = cv2.cvtColor(shirt_overlay, cv2.COLOR_BGR2BGRA)  # Ensure alpha channel

        # Perform the overlay
        roi_bg = (roi * mask_inv[:, :, np.newaxis]).astype(np.uint8)
        shirt_fg = (shirt_overlay[:, :, :3] * mask[:, :, np.newaxis]).astype(np.uint8)

        frame[y1:y2, x1:x2] = cv2.add(roi_bg, shirt_fg)

    except Exception as e:
        print(f"Error in overlay_shirt: {e}")

    return frame

def generate_frames():
    global current_shirt

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if current_shirt is not None:
                frame = overlay_shirt(frame, current_shirt, results.pose_landmarks.landmark)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_shirts_from_prompt(prompt):
    global generated_shirts

    search_query = f"{prompt} only want 3d images! ,men's shirt, high resolution, single shirt image only, only image must be there in a photo, no body parts or backgrounds, isolated shirt, high-definition, full size image, focused on fabric texture and details, no logos or distractions"
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

                # Remove background and auto-crop
                shirt_img = remove_background(img_url)

                if shirt_img is not None:
                    generated_shirts.append({
                        'id': i,
                        'url': img_url,
                        'image': shirt_img
                    })
            except requests.exceptions.RequestException as e:
                print(f"Network error processing image {i}: {e}")
            except Exception as e:
                print(f"Error processing image {i}: {e}")

        return True, "Shirts generated successfully"
    except requests.exceptions.RequestException as e:
        return False, f"API network error: {str(e)}"
    except Exception as e:
        return False, f"API error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html', shirts=generated_shirts)

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

# Handle shirt selection via socket
@socketio.on('select_shirt')
def handle_select_shirt(data):
    global current_shirt
    shirt_url = data.get('shirt_url')

    if shirt_url:
        # Remove background and prepare shirt
        shirt_img = remove_background(shirt_url)
        if shirt_img is not None:
            current_shirt = shirt_img
            print("[INFO] Shirt selected and processed successfully.")
            socketio.emit('shirt_selected', {'status': 'success'})
        else:
            print("[ERROR] Failed to process the shirt image.")
            socketio.emit('shirt_selected', {'status': 'error'})
    else:
        print("[ERROR] No shirt URL provided.")
        socketio.emit('shirt_selected', {'status': 'error'})


@app.route('/select_shirt', methods=['POST'])
def select_shirt():
    global current_shirt
    shirt_id = int(request.form.get('shirt_id'))

    if 0 <= shirt_id < len(generated_shirts):
        current_shirt = generated_shirts[shirt_id]['image']
        return jsonify({'success': True, 'message': f'Shirt {shirt_id} selected'})

    return jsonify({'success': False, 'message': 'Invalid shirt ID'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')