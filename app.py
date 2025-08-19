import os
import cv2
import time
import json
import base64
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, Response, jsonify, make_response
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix

# ---- Optional: eventlet monkey patch for best SocketIO behavior ----
try:
    import eventlet  # type: ignore
    eventlet.monkey_patch()
except Exception:
    pass

# ------------------------ CONFIG ------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "YOUR_SERPAPI_KEY")
REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY", "YOUR_REMOVEBG_KEY")

# If server has no webcam, set this to -1 to disable capture.
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# -------------------- MediaPipe (pose) -------------------
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# -------------------- App Factory ------------------------
def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    # Honor X-Forwarded-* from Cloudflare/Cloudflared
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    socketio = SocketIO(
        app,
        async_mode="eventlet",
        cors_allowed_origins="*",
        ping_interval=20,
        ping_timeout=30,
        logger=False,
        engineio_logger=False
    )

    # Runtime state
    state = {
        "current_shirt": None,
        "generated_shirts": [],
        "current_fit_score": 0.0
    }

    # Try to open camera once (don’t crash if headless)
    cap = cv2.VideoCapture(CAMERA_INDEX) if CAMERA_INDEX >= 0 else None

    # ----------------- Utility functions -----------------
    def calculate_fit_score(landmarks, shirt_img, frame_shape):
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            shoulder_width = np.linalg.norm(
                np.array([left_shoulder.x * frame_shape[1], left_shoulder.y * frame_shape[0]]) -
                np.array([right_shoulder.x * frame_shape[1], right_shoulder.y * frame_shape[0]])
            )
            torso_length = np.linalg.norm(
                np.array([left_shoulder.x * frame_shape[1], left_shoulder.y * frame_shape[0]]) -
                np.array([left_hip.x * frame_shape[1], left_hip.y * frame_shape[0]])
            )

            shirt_h, shirt_w = shirt_img.shape[:2]
            width_ratio = min(shoulder_width, shirt_w) / max(shoulder_width, shirt_w)
            height_ratio = min(torso_length, shirt_h) / max(torso_length, shirt_h)

            shoulder_angle = np.degrees(np.arctan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ))
            angle_score = max(0, 1 - abs(shoulder_angle) / 45)

            fit_score = (width_ratio * 0.5 + height_ratio * 0.3 + angle_score * 0.2) * 100
            fit_score = max(0, min(100, float(fit_score)))
            return fit_score
        except Exception:
            return 50.0

    def auto_crop_image(img):
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            coords = cv2.findNonZero(alpha)
            if coords is None:
                return img
            x, y, w, h = cv2.boundingRect(coords)
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(img.shape[1] - x, w + 2 * pad_x)
            h = min(img.shape[0] - y, h + 2 * pad_y)
            return img[y:y+h, x:x+w]
        return img

    def remove_background(image_url):
        """Download + send to remove.bg; return BGRA numpy or None."""
        try:
            r = requests.get(image_url, timeout=15)
            if r.status_code != 200:
                print(f"Download error: {r.status_code}")
                return None

            # remove.bg expects a filename; many proxies reject nameless multipart
            files = {
                "image_file": ("input.png", r.content, "image/png")
            }
            data = {"size": "auto"}
            headers = {"X-Api-Key": REMOVE_BG_API_KEY}

            rb = requests.post(
                "https://api.remove.bg/v1.0/removebg",
                files=files,
                data=data,
                headers=headers,
                timeout=60
            )
            if rb.status_code != 200:
                print(f"remove.bg error {rb.status_code}: {rb.text[:200]}")
                return None

            img = Image.open(BytesIO(rb.content))
            # Apply EXIF orientation if present
            try:
                exif = img.getexif()
                orientation = exif.get(274)
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
            except Exception:
                pass

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
            return auto_crop_image(img_cv)
        except Exception as e:
            print(f"remove_background exception: {e}")
            return None

    def calculate_shirt_position(landmarks, frame_shape):
        try:
            ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            scx = (ls.x + rs.x) / 2
            scy = (ls.y + rs.y) / 2
            hcx = (lh.x + rh.x) / 2
            hcy = (lh.y + rh.y) / 2

            torso_len_norm = np.linalg.norm(np.array([scx, scy]) - np.array([hcx, hcy]))
            torso_len_px = int(torso_len_norm * frame_shape[0] * 1.5)

            shoulder_width_px = int(
                np.linalg.norm(
                    np.array([ls.x * frame_shape[1], ls.y * frame_shape[0]]) -
                    np.array([rs.x * frame_shape[1], rs.y * frame_shape[0]])
                ) * 1.3
            )

            shirt_w = int(shoulder_width_px * 1.1)
            shirt_h = int(torso_len_px * 1.2)

            center_x = int(scx * frame_shape[1])
            center_y = int((scy * 0.6 + hcy * 0.4) * frame_shape[0])

            if rs.x > ls.x:
                angle = np.degrees(np.arctan2(rs.y - ls.y, rs.x - ls.x))
            else:
                angle = np.degrees(np.arctan2(ls.y - rs.y, ls.x - rs.x))

            return {"width": shirt_w, "height": shirt_h, "center_x": center_x, "center_y": center_y, "angle": angle}
        except Exception as e:
            print(f"calculate_shirt_position error: {e}")
            return None

    def overlay_shirt(frame, shirt_img, landmarks):
        if shirt_img is None:
            return frame

        try:
            pos = calculate_shirt_position(landmarks, frame.shape)
            if pos is None:
                return frame

            s_h, s_w = shirt_img.shape[:2]
            scale = max(pos["width"] / s_w, pos["height"] / s_h)
            new_w = int(s_w * scale)
            new_h = int(s_h * scale)

            shirt_resized = cv2.resize(shirt_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rot = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), pos["angle"], 1)
            shirt_rot = cv2.warpAffine(shirt_resized, rot, (new_w, new_h), borderMode=cv2.BORDER_TRANSPARENT)

            x_off = pos["center_x"] - new_w // 2
            y_off = pos["center_y"] - new_h // 3

            y1, y2 = max(0, y_off), min(frame.shape[0], y_off + new_h)
            x1, x2 = max(0, x_off), min(frame.shape[1], x_off + new_w)

            roi = frame[y1:y2, x1:x2]
            shirt_overlay = shirt_rot[max(0, -y_off):min(new_h, frame.shape[0] - y_off),
                                      max(0, -x_off):min(new_w, frame.shape[1] - x_off)]

            if shirt_overlay.shape[2] == 4:
                mask = shirt_overlay[:, :, 3] / 255.0
            else:
                mask = np.ones(shirt_overlay.shape[:2], dtype=np.float32)
                shirt_overlay = cv2.cvtColor(shirt_overlay, cv2.COLOR_BGR2BGRA)

            mask_inv = 1.0 - mask
            roi_bg = (roi * mask_inv[:, :, None]).astype(np.uint8)
            shirt_fg = (shirt_overlay[:, :, :3] * mask[:, :, None]).astype(np.uint8)
            frame[y1:y2, x1:x2] = cv2.add(roi_bg, shirt_fg)

            # Fit score + emit over websocket
            state["current_fit_score"] = calculate_fit_score(landmarks, shirt_img, frame.shape)
            cv2.putText(frame, f"Fit: {int(state['current_fit_score'])}%", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            socketio.emit("fit_score_update", {"score": state["current_fit_score"]})
        except Exception as e:
            print(f"overlay_shirt error: {e}")
        return frame

    # ----------------------- Routes -----------------------
    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/")
    def index():
        return render_template("index.html", shirts=state["generated_shirts"])

    def frame_generator():
        """MJPEG generator. Works through Cloudflare when headers are right."""
        if cap is None or not cap.isOpened():
            # Produce a black frame with message so the stream doesn't 400
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera unavailable", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", blank)
            chunk = buf.tobytes()
            while True:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Cache-Control: no-cache\r\n\r\n" + chunk + b"\r\n")
                time.sleep(0.1)

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if state["current_shirt"] is not None:
                    frame = overlay_shirt(frame, state["current_shirt"], results.pose_landmarks.landmark)

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-cache\r\n\r\n" + buf.tobytes() + b"\r\n")

    @app.route("/video_feed")
    def video_feed():
        resp = Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
        # Extra headers reduce buffering/proxy issues
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Accel-Buffering"] = "no"  # some proxies respect this
        return resp

    @app.route("/generate_shirts", methods=["POST"])
    def generate_shirts():
        prompt = (request.form.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"success": False, "message": "Please enter a prompt"}), 400

        search_query = (
            f"{prompt} only — isolated 3D high-resolution image, full-size, centered, "
            "no background, no body parts, no models, no props, no logos — just the clothing item. "
            "Fabric texture, folds, and details must be sharp and clear. Focus on realism and textile quality. "
            "Clothing item should be cleanly cropped and well-lit, studio-style"
        )

        params = {
            "engine": "google",
            "q": search_query,
            "tbm": "isch",
            "api_key": SERPAPI_KEY,
        }

        try:
            r = requests.get("https://serpapi.com/search", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return jsonify({"success": False, "message": f"Search error: {e}"}), 502

        state["generated_shirts"].clear()
        images = data.get("images_results", [])[:4]

        for i, image in enumerate(images):
            try:
                img_url = image.get("original")
                if not img_url:
                    continue
                shirt_img = remove_background(img_url)
                if shirt_img is not None:
                    state["generated_shirts"].append({"id": i, "url": img_url, "image": shirt_img})
            except Exception as e:
                print(f"Image {i} processing error: {e}")

        # Return base64 for client preview
        payload = []
        for shirt in state["generated_shirts"]:
            _, buf = cv2.imencode(".png", shirt["image"])
            payload.append({"id": shirt["id"], "url": shirt["url"], "image": base64.b64encode(buf).decode("utf-8")})

        return jsonify({"success": True, "shirts": payload})

    @app.route("/select_shirt", methods=["POST"])
    def select_shirt():
        try:
            shirt_id = int(request.form.get("shirt_id", "-1"))
        except ValueError:
            return jsonify({"success": False, "message": "Invalid shirt ID"}), 400

        if 0 <= shirt_id < len(state["generated_shirts"]):
            state["current_shirt"] = state["generated_shirts"][shirt_id]["image"]
            return jsonify({"success": True, "message": f"Shirt {shirt_id} selected"})
        return jsonify({"success": False, "message": "Invalid shirt ID"}), 400

    # ------------------- SocketIO events ------------------
    @socketio.on("select_shirt")
    def ws_select_shirt(data):
        url = (data or {}).get("shirt_url")
        if not url:
            socketio.emit("shirt_selected", {"status": "error", "reason": "no_url"})
            return
        img = remove_background(url)
        if img is None:
            socketio.emit("shirt_selected", {"status": "error"})
            return
        state["current_shirt"] = img
        socketio.emit("shirt_selected", {"status": "success"})

    # ---------------- Error Handlers (nice 400s) ----------
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify(error="bad_request", detail=str(e)), 400

    @app.errorhandler(502)
    def bad_gateway(e):
        return jsonify(error="bad_gateway", detail=str(e)), 502

    return app, socketio


# -------------- Dev entrypoint (optional) ---------------
app, socketio = create_app()

if __name__ == "__main__":
    # For quick local runs. In production use gunicorn (see below).
    socketio.run(app, host="0.0.0.0", port=8000)