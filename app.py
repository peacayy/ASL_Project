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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_and_random_key_that_should_be_changed')

asl_recognizer_instance = None
camera_instance = None
prediction_lock = threading.Lock()
shared_current_prediction = ""
shared_prediction_confidence = 0.0

class ASLRecognizer:
    def __init__(self, model_path, label_encoder_path, config_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded: {model_path}")
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Label encoder loaded: {label_encoder_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.sequence_length = self.config.get('sequence_length', 10)
            self.class_names = self.config.get('class_names', [])
            logger.info(f"Config: sequence_length={self.sequence_length}, classes={len(self.class_names)}")
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.sequence_buffer = deque(maxlen=self.sequence_length)
            self.prediction_threshold = 0.7
            logger.info("ASLRecognizer initialized")
        except Exception as e:
            logger.error(f"Error initializing ASLRecognizer: {e}", exc_info=True)
            raise

    def extract_landmarks(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks, dtype=np.float32)
            return np.zeros(63, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error extracting landmarks: {e}")
            return np.zeros(63, dtype=np.float32)

    def normalize_landmarks(self, landmarks):
        if landmarks.sum() == 0:
            return np.zeros(63, dtype=np.float32)
        reshaped = landmarks.reshape(21, 3)
        wrist = reshaped[0]
        normalized = reshaped - wrist
        return normalized.flatten()

    def predict_from_frame(self, frame):
        try:
            landmarks = self.extract_landmarks(frame)
            normalized_landmarks = self.normalize_landmarks(landmarks)
            self.sequence_buffer.append(normalized_landmarks)
            
            if len(self.sequence_buffer) == self.sequence_length:
                sequence_input = np.array(list(self.sequence_buffer), dtype=np.float32)
                sequence_input = np.expand_dims(sequence_input, axis=0)
                prediction_probs = self.model.predict(sequence_input, verbose=0)[0]
                predicted_class_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_class_idx]
                if confidence > self.prediction_threshold:
                    predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                    logger.info(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
                    return predicted_class, confidence
            return None, 0.0
        except Exception as e:
            logger.error(f"Error in predict_from_frame: {e}")
            return None, 0.0

    def predict_from_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video: {video_path}")
            
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            predictions_list = []
            self.sequence_buffer.clear()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_skip_interval = max(1, int(frame_rate / 10))
                if frame_count % frame_skip_interval == 0:
                    prediction, confidence = self.predict_from_frame(frame.copy())
                    if prediction:
                        predictions_list.append(prediction)
                frame_count += 1
            
            cap.release()
            
            if predictions_list:
                from collections import Counter
                most_common_sign = Counter(predictions_list).most_common(1)[0][0]
                logger.info(f"Video processed: {most_common_sign}")
                return most_common_sign, 1.0
            logger.info(f"No signs detected in video: {video_path}")
            return None, 0.0
        except Exception as e:
            logger.error(f"Error processing video '{video_path}': {e}")
            return None, 0.0

def initialize_application():
    global asl_recognizer_instance
    try:
        model_path = os.path.join("model_outputs", "final_asl_model.h5")
        label_encoder_path = os.path.join("model_outputs", "label_encoder.pkl")
        config_path = os.path.join("model_outputs", "model_config.json")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.join('static', 'audio'), exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
        asl_recognizer_instance = ASLRecognizer(model_path, label_encoder_path, config_path)
        logger.info("Application initialized")
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        exit(1)

def text_to_speech(text, filename_prefix):
    try:
        audio_dir = os.path.join('static', 'audio')
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '_')).strip() or "unknown"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        audio_filename = f'{filename_prefix}_{safe_text}_{timestamp}.mp3'
        audio_path_full = os.path.join(audio_dir, audio_filename)
        logger.info(f"Saving audio to: {os.path.abspath(audio_path_full)}")
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path_full)
        
        if os.path.exists(audio_path_full):
            logger.info(f"Audio saved: {audio_path_full} ({os.path.getsize(audio_path_full)} bytes)")
            return audio_filename
        logger.error(f"Audio file not created: {audio_path_full}")
        return None
    except Exception as e:
        logger.error(f"Error generating speech for '{text}': {e}", exc_info=True)
        return None

def generate_frames():
    global camera_instance, shared_current_prediction, shared_prediction_confidence
    if camera_instance is None:
        camera_instance = cv2.VideoCapture(0)
        if not camera_instance.isOpened():
            logger.error("Failed to open webcam")
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + b'Error: Could not open webcam' + b'\r\n')
            return
        camera_instance.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("Webcam initialized")

    if asl_recognizer_instance:
        asl_recognizer_instance.sequence_buffer.clear()
        logger.info("Sequence buffer cleared")

    try:
        while True:
            success, frame = camera_instance.read()
            if not success:
                logger.warning("Failed to read webcam frame")
                break
            frame = cv2.flip(frame, 1)
            
            if asl_recognizer_instance:
                prediction, confidence = asl_recognizer_instance.predict_from_frame(frame.copy())
                with prediction_lock:
                    if prediction:
                        shared_current_prediction = prediction
                        shared_prediction_confidence = confidence
                        logger.debug(f"Updated prediction: {prediction} (Confidence: {confidence:.2f})")
                    else:
                        shared_current_prediction = ""
                        shared_prediction_confidence = 0.0

            display_prediction = shared_current_prediction if shared_current_prediction else "Detecting..."
            cv2.putText(frame, f"Sign: {display_prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if shared_prediction_confidence > 0:
                cv2.putText(frame, f"Confidence: {shared_prediction_confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if camera_instance:
            camera_instance.release()
            logger.info("Webcam released")
            camera_instance = None

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
            video_file = request.files['video']
            allowed_extensions = {'mp4', 'avi', 'mov'}
            if '.' in video_file.filename and video_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                try:
                    video_filename = secure_filename(f"uploaded_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
                    video_path = os.path.join('uploads', video_filename)
                    video_file.save(video_path)
                    logger.info(f"Uploaded video saved: {video_path}")

                    if asl_recognizer_instance:
                        prediction, confidence = asl_recognizer_instance.predict_from_video(video_path)
                        if prediction:
                            text_result = prediction.upper()
                            audio_filename = text_to_speech(text_result, "video_prediction")
                            flash(f"Recognized: {text_result}", 'success')
                        else:
                            error_message = "No clear sign detected in video"
                            flash(error_message, 'warning')
                    else:
                        error_message = "Recognition model not loaded"
                        flash(error_message, 'error')
                except Exception as e:
                    error_message = f"Error processing video: {str(e)}"
                    flash(error_message, 'error')
                finally:
                    if 'video_path' in locals() and os.path.exists(video_path):
                        os.remove(video_path)
                        logger.info(f"Cleaned up video: {video_path}")
            else:
                error_message = "Invalid video format. Use MP4, AVI, or MOV."
                flash(error_message, 'warning')

        elif request.form.get('action') == 'predict':
            with prediction_lock:
                if shared_current_prediction and shared_prediction_confidence > 0.5:
                    text_result = shared_current_prediction.upper()
                    audio_filename = text_to_speech(text_result, "webcam_capture")
                    if audio_filename:
                        flash(f"Recognized: {text_result}", 'success')
                        shared_current_prediction = ""
                        shared_prediction_confidence = 0.0
                    else:
                        error_message = "Failed to generate audio"
                        flash(error_message, 'warning')
                else:
                    error_message = "No clear sign detected from webcam"
                    flash(error_message, 'warning')
            
            response_html = f'''<!DOCTYPE html>
            <html>
            <body>
                <div id="text-result">{text_result}</div>
                <div id="audio-result">{audio_filename if audio_filename else ''}</div>
                <div class="error-message">{error_message if error_message else ''}</div>
            </body>
            </html>'''
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
        name = request.form.get('name')
        email = request.form.get('email')
        remark = request.form.get('remark')
        logger.info(f"Feedback from {name} ({email}): {remark}")
        flash("Thank you for your feedback!", 'success')
        return redirect(url_for('feedback'))
    return render_template('feedback.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    initialize_application()
    app.run(debug=True, host='0.0.0.0', port=5000)
