import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import json
from gtts import gTTS
import threading
import logging
from datetime import datetime
from collections import deque
from flask import Flask, render_template, request, Response, url_for, jsonify, flash, redirect
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_and_random_key')

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'aslconverter@gmail.com'  # Replace with sender email
app.config['MAIL_PASSWORD'] = os.environ.get('njpybkbaqvqinhlu')  # Set in environment
app.config['MAIL_DEFAULT_SENDER'] = 'aslconverter@gmail.com'

mail = Mail(app)

from flask import Flask, render_template, request, flash, redirect, url_for
import smtplib
from email.message import EmailMessage

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Replace with your actual Gmail and app password
SENDER_EMAIL = 'aslconverter@gmail.com'
SENDER_PASSWORD = 'njpybkbaqvqinhlu'


asl_recognizer_instance = None
camera_instance = None
prediction_lock = threading.Lock()
shared_current_prediction = ""
shared_prediction_confidence = 0.0

class ASLRecognizer:
    def __init__(self, model_path, label_encoder_path, config_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.sequence_length = self.config.get('sequence_length', 10)
        self.class_names = self.config.get('class_names', [])
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_threshold = 0.7

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks, dtype=np.float32)
        return np.zeros(63, dtype=np.float32)

    def normalize_landmarks(self, landmarks):
        if landmarks.sum() == 0:
            return np.zeros(63, dtype=np.float32)
        reshaped = landmarks.reshape(21, 3)
        wrist = reshaped[0]
        normalized = reshaped - wrist
        return normalized.flatten()

    def predict_from_frame(self, frame):
        landmarks = self.extract_landmarks(frame)
        normalized = self.normalize_landmarks(landmarks)
        self.sequence_buffer.append(normalized)

        if len(self.sequence_buffer) == self.sequence_length:
            sequence_input = np.expand_dims(np.array(self.sequence_buffer), axis=0)
            prediction_probs = self.model.predict(sequence_input, verbose=0)[0]
            predicted_idx = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_idx]
            if confidence > self.prediction_threshold:
                return self.label_encoder.inverse_transform([predicted_idx])[0], confidence
        return None, 0.0

    def predict_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        predictions = []
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % max(1, int(frame_rate / 10)) == 0:
                pred, conf = self.predict_from_frame(frame)
                if pred:
                    predictions.append(pred)
            frame_count += 1
        cap.release()
        if predictions:
            from collections import Counter
            return Counter(predictions).most_common(1)[0][0], 1.0
        return None, 0.0

def initialize_application():
    global asl_recognizer_instance
    model_path = os.path.join("model_outputs", "final_asl_model.h5")
    label_path = os.path.join("model_outputs", "label_encoder.pkl")
    config_path = os.path.join("model_outputs", "model_config.json")
    asl_recognizer_instance = ASLRecognizer(model_path, label_path, config_path)

    os.makedirs(os.path.join('static', 'audio'), exist_ok=True)
    os.makedirs('uploads', exist_ok=True)

def text_to_speech(text, prefix):
    audio_dir = os.path.join('static', 'audio')
    safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '_')).strip() or "unknown"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f"{prefix}_{safe_text}_{timestamp}.mp3"
    path = os.path.join(audio_dir, filename)
    gTTS(text=text, lang='en').save(path)
    return filename

def generate_frames():
    global camera_instance, shared_current_prediction, shared_prediction_confidence
    if camera_instance is None:
        camera_instance = cv2.VideoCapture(0)
        camera_instance.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    asl_recognizer_instance.sequence_buffer.clear()

    while True:
        success, frame = camera_instance.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        pred, conf = asl_recognizer_instance.predict_from_frame(frame.copy())
        with prediction_lock:
            if pred:
                shared_current_prediction = pred
                shared_prediction_confidence = conf
            else:
                shared_current_prediction = ""
                shared_prediction_confidence = 0.0

        display_text = shared_current_prediction if shared_current_prediction else "Detecting..."
        cv2.putText(frame, f"Sign: {display_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if shared_prediction_confidence > 0:
            cv2.putText(frame, f"Confidence: {shared_prediction_confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/conversion', methods=['GET', 'POST'])
def conversion():
    text_result = ""
    audio_filename = None
    error_message = None

    if request.method == 'POST':
        if 'video' in request.files and request.files['video'].filename:
            video = request.files['video']
            filename = secure_filename(f"uploaded_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
            path = os.path.join('uploads', filename)
            video.save(path)

            pred, conf = asl_recognizer_instance.predict_from_video(path)
            os.remove(path)
            if pred:
                text_result = pred.upper()
                audio_filename = text_to_speech(text_result, "video")
                flash(f"Recognized: {text_result}", 'success')
            else:
                flash("No sign detected", 'warning')
        elif request.form.get('action') == 'predict':
            with prediction_lock:
                if shared_current_prediction and shared_prediction_confidence > 0.5:
                    text_result = shared_current_prediction.upper()
                    audio_filename = text_to_speech(text_result, "webcam")
                    flash(f"Recognized: {text_result}", 'success')
                    shared_current_prediction = ""
                    shared_prediction_confidence = 0.0
                else:
                    flash("No clear sign detected", 'warning')

        response_html = f'''
        <!DOCTYPE html>
        <html><body>
        <div id="text-result">{text_result}</div>
        <div id="audio-result">{audio_filename or ''}</div>
        </body></html>'''
        return response_html

    audio_url = url_for('static', filename=f'audio/{audio_filename}') if audio_filename else None
    return render_template('conversion.html', text=text_result, audio=audio_url, error=error_message)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_prediction')
def current_prediction_api():
    with prediction_lock:
        return jsonify({
            'prediction': shared_current_prediction,
            'confidence': float(shared_prediction_confidence)
        })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        user_name = request.form.get('name')
        user_email = request.form.get('email')
        user_message = request.form.get('remark') 

        if not user_name or not user_email or not user_message:
            flash('Please fill in all fields.', 'warning')
            return redirect(url_for('feedback'))

        msg = EmailMessage()
        msg['Subject'] = f'Feedback from {user_name}'
        msg['From'] = user_email  # <-- use user's entered email as From
        msg['To'] = 'aslconverter@gmail.com'  # Your email where feedback goes

        body = f"Name: {user_name}\nEmail: {user_email}\n\nMessage:\n{user_message}"
        msg.set_content(body)

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                smtp.send_message(msg)
            flash('Thank you for your feedback!', 'success')
        except Exception as e:
            print(f"Error sending email: {e}")
            flash('Failed to send feedback. Please try again later.', 'danger')

        return redirect(url_for('feedback'))

    return render_template('feedback.html')



@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    initialize_application()
    app.run(debug=True, host='0.0.0.0', port=5000)
