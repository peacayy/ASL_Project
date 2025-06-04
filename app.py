import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import json
from gtts import gTTS
import threading
import time
from datetime import datetime
import logging
from collections import deque # Import deque for efficient buffer

from flask import Flask, render_template, request, jsonify, Response, url_for, redirect, flash

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# IMPORTANT: Change this to a strong, random key for production!
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_and_random_key_that_should_be_changed')

# --- Global objects and state for the application ---
# These will be initialized once when the app starts
asl_recognizer_instance = None # Renamed to avoid confusion with the class name
camera_instance = None

# Shared state for real-time webcam prediction results
# Using a Lock for thread-safe access to shared variables
prediction_lock = threading.Lock()
shared_current_prediction = ""
shared_prediction_confidence = 0.0

# --- ASLRecognizer Class ---
class ASLRecognizer:
    def __init__(self, model_path, label_encoder_path, config_path):
        """Initialize ASL recognizer with trained model and MediaPipe."""
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path} successfully.")

            # Load label encoder
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Label encoder loaded from {label_encoder_path} successfully.")

            # Load configuration
            with open(config_path, 'r') as f:
                self.config = json.load(f)

            self.sequence_length = self.config.get('sequence_length', 10) # Default to 10
            self.class_names = self.config.get('class_names', [])
            logger.info(f"Model config loaded. Sequence length: {self.sequence_length}, Classes: {len(self.class_names)}")

            # Initialize MediaPipe Hands
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, # Crucial for real-time video tracking
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )

            # Use deque for efficient sequence buffering
            self.sequence_buffer = deque(maxlen=self.sequence_length)
            self.prediction_threshold = 0.7 # Adjusted for potentially better responsiveness (was 0.8)

            logger.info("ASL Recognizer initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing ASL Recognizer: {e}")
            raise # Re-raise to ensure app doesn't start if critical components fail

    def extract_landmarks(self, image):
        """Extract hand landmarks from image using MediaPipe."""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks, dtype=np.float32) # Ensure float32 for consistency
            else:
                return np.zeros(63, dtype=np.float32) # Return zeros if no hand detected

        except Exception as e:
            logger.warning(f"Error extracting landmarks: {e}. Returning zeros.")
            return np.zeros(63, dtype=np.float32) # Return zeros on error

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position."""
        if landmarks.sum() == 0: # Check if it's an array of zeros (no hand detected)
            return np.zeros(63, dtype=np.float32)

        # Reshape to (21, 3) for easier manipulation
        reshaped = landmarks.reshape(21, 3)

        # Use wrist (landmark 0) as reference point
        wrist = reshaped[0]

        # Normalize all landmarks relative to the wrist
        normalized = reshaped - wrist

        # Flatten back to 1D array
        return normalized.flatten()

    def predict_from_frame(self, frame):
        """
        Processes a single frame for landmarks, updates the sequence buffer,
        and makes a prediction if the buffer is full and confidence is high.
        """
        try:
            # Extract and normalize landmarks
            landmarks = self.extract_landmarks(frame)
            normalized_landmarks = self.normalize_landmarks(landmarks)

            # Add to sequence buffer (deque handles maxlen automatically)
            self.sequence_buffer.append(normalized_landmarks)

            predicted_class = None
            confidence = 0.0

            # Make prediction only if buffer is full
            if len(self.sequence_buffer) == self.sequence_length:
                # Prepare input for model: (1, sequence_length, num_features)
                sequence_input = np.array(list(self.sequence_buffer), dtype=np.float32)
                sequence_input = np.expand_dims(sequence_input, axis=0) # Add batch dimension

                # Make prediction
                prediction_probs = self.model.predict(sequence_input, verbose=0)[0]
                predicted_class_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_class_idx]

                # Only return prediction if confidence is above threshold
                if confidence > self.prediction_threshold:
                    predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]

            return predicted_class, confidence

        except Exception as e:
            logger.error(f"Error in real-time prediction from frame: {e}")
            return None, 0.0

    def predict_from_video(self, video_path):
        """Predict ASL signs from an uploaded video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {video_path}")

            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Processing video '{video_path}' with frame rate: {frame_rate} FPS")

            predictions_list = []
            
            # Reset sequence buffer for video processing
            self.sequence_buffer.clear() 

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # End of video or error

                # Process every Xth frame to reduce computation for uploaded videos
                # Adjust 'frame_skip_interval' based on video length and desired accuracy
                frame_skip_interval = max(1, int(frame_rate / 10)) # Aim for ~10 predictions per second
                if frame_count % frame_skip_interval == 0:
                    # Using the predict_from_frame logic to fill and use the sequence buffer
                    prediction, confidence = self.predict_from_frame(frame.copy())
                    if prediction: # Only append if a clear prediction is made
                        predictions_list.append(prediction)

                frame_count += 1
            
            cap.release()

            # Determine the most common prediction from the video
            if predictions_list:
                from collections import Counter
                most_common_sign = Counter(predictions_list).most_common(1)[0][0]
                
                # Optionally calculate average confidence for the most common sign
                # (This requires storing confidences along with predictions_list)
                # For simplicity now, we just return the most common sign.
                # If you need avg_confidence, you'll need to modify predictions_list to store tuples (sign, confidence)
                logger.info(f"Video '{video_path}' processed. Most common sign: {most_common_sign}")
                return most_common_sign, 1.0 # Return 1.0 as placeholder for confidence for now
            
            logger.info(f"No clear signs detected in video '{video_path}'.")
            return None, 0.0 # No clear prediction from the video

        except Exception as e:
            logger.error(f"Error processing video '{video_path}': {e}")
            return None, 0.0

# --- Initialization Function ---
def initialize_application_components():
    """Initializes the ASL Recognizer and creates necessary directories."""
    global asl_recognizer_instance, camera_instance
    logger.info("Initializing ASL Recognition application components...")

    try:
        model_path = "model_outputs/final_asl_model.h5"
        label_encoder_path = "model_outputs/label_encoder.pkl"
        config_path = "model_outputs/model_config.json"

        # Create model_outputs directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        asl_recognizer_instance = ASLRecognizer(model_path, label_encoder_path, config_path)
        logger.info("ASL Recognizer instance created successfully.")

        # Initialize camera here if you want it ready from the start
        # However, it's often better to initialize it when generate_frames is first called
        # to avoid resource contention if not immediately used.
        # For this setup, initializing in generate_frames() is fine.

        # Create directories for uploads and audio files
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('static/audio', exist_ok=True)
        logger.info("Uploads and static/audio directories ensured.")

    except Exception as e:
        logger.error(f"Critical error during application initialization: {e}")
        # Optionally, flash a message to the user or redirect to an error page
        # In a real app, you might want to display a maintenance page
        exit(1) # Exit if essential components fail to load


# --- Video Frame Generator for Webcam ---
def generate_frames():
    """
    Generator function that captures webcam frames, performs real-time ASL prediction,
    draws overlays, and yields JPEG frames for Flask's video stream.
    """
    global camera_instance, shared_current_prediction, shared_prediction_confidence

    if camera_instance is None:
        camera_instance = cv2.VideoCapture(0) # Open the default webcam
        if not camera_instance.isOpened():
            logger.error("Failed to open webcam. Please ensure it's connected and not in use.")
            # Yield an error message to the client if camera fails
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + b'Error: Could not open webcam. Please ensure it is connected and not in use.' + b'\r\n')
            return # Exit the generator

        camera_instance.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("Webcam initialized.")

    # Reset the recognizer's internal sequence buffer for a fresh start with webcam stream
    if asl_recognizer_instance:
        asl_recognizer_instance.sequence_buffer.clear()
        logger.info("ASL Recognizer sequence buffer cleared for new webcam session.")

    while True:
        success, frame = camera_instance.read()
        if not success:
            logger.warning("Failed to read frame from webcam. Stream interrupted.")
            break # Exit loop if frame reading fails (e.g., camera disconnected)

        frame = cv2.flip(frame, 1) # Mirror effect for natural webcam view

        # --- Real-time Prediction ---
        if asl_recognizer_instance:
            prediction, confidence = asl_recognizer_instance.predict_from_frame(frame.copy()) # Pass a copy to avoid modification issues
            
            with prediction_lock: # Protect access to shared variables
                if prediction:
                    # Implement simple debouncing: only update if the prediction changes or if it's the first prediction
                    if prediction != shared_current_prediction:
                        shared_current_prediction = prediction
                        shared_prediction_confidence = confidence
                        logger.debug(f"Live Prediction: {prediction} (Confidence: {confidence:.2f})")
                else: # No confident prediction, reset or set to empty
                    shared_current_prediction = "" # Or "No sign"
                    shared_prediction_confidence = 0.0

        # --- Overlay Prediction on Frame ---
        display_prediction = shared_current_prediction if shared_current_prediction else "Detecting..."
        display_confidence = shared_prediction_confidence
        
        cv2.putText(frame, f"Sign: {display_prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if display_confidence > 0: # Only show confidence if there's a prediction
            cv2.putText(frame, f"Confidence: {display_confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Release camera resources when the generator exits
    if camera_instance:
        camera_instance.release()
        logger.info("Webcam released.")
        camera_instance = None # Reset global variable

# --- Text-to-Speech Function ---
def text_to_speech(text, filename_prefix):
    """
    Converts text to speech using gTTS and saves it as an MP3 file.
    Returns the web-accessible filename.
    """
    try:
        # Sanitize text for filename if needed
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '_')).strip()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # Milliseconds for more uniqueness
        audio_filename = f'{filename_prefix}_{safe_text}_{timestamp}.mp3'
        audio_path_full = os.path.join('static', 'audio', audio_filename)

        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path_full)

        logger.info(f"Audio saved: {audio_path_full}")
        return url_for('static', filename=f'audio/{audio_filename}', _external=True) # Return URL

    except Exception as e:
        logger.error(f"Error generating speech for '{text}': {e}")
        return None

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('home.html')

@app.route('/conversion', methods=['GET', 'POST'])
def conversion():
    """
    Handles conversion page logic, including video uploads and triggering webcam prediction.
    """
    text_result = ""
    audio_file_url = None
    error_message = None

    if request.method == 'POST':
        if 'video' in request.files:
            # --- Handle Video Upload ---
            video_file = request.files['video']
            if video_file.filename == '':
                flash('No selected video file.', 'warning')
                return redirect(request.url)

            if asl_recognizer_instance is None:
                flash("Recognition model not loaded. Please try again later.", 'error')
                return redirect(request.url)

            try:
                # Secure filename and save
                # For a real application, you'd use werkzeug.utils.secure_filename
                upload_dir = 'uploads'
                os.makedirs(upload_dir, exist_ok=True)
                video_filename = f"uploaded_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4" # Enforce .mp4
                video_path = os.path.join(upload_dir, video_filename)
                video_file.save(video_path)
                logger.info(f"Uploaded video saved to: {video_path}")

                # Process video
                prediction, confidence = asl_recognizer_instance.predict_from_video(video_path)
                
                if prediction and confidence > 0: # confidence > 0 implies a prediction was made
                    text_result = prediction.upper()
                    audio_file_url = text_to_speech(text_result, "video_prediction")
                    flash(f"Video processed successfully! Recognized: {text_result}", 'success')
                else:
                    error_message = "Could not detect any clear sign in the uploaded video."
                    flash(error_message, 'warning')

            except Exception as e:
                logger.error(f"Error processing uploaded video: {e}")
                error_message = f"An error occurred while processing the video: {str(e)}"
                flash(error_message, 'error')
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path) # Clean up uploaded file
                    logger.info(f"Cleaned up uploaded video: {video_path}")

        elif request.form.get('action') == 'predict_webcam_to_text':
            # --- Handle Webcam Prediction to Text/Speech ---
            with prediction_lock:
                if shared_current_prediction and shared_prediction_confidence > 0.5: # Use shared state
                    text_result = shared_current_prediction.upper()
                    audio_file_url = text_to_speech(text_result, "webcam_capture")
                    flash(f"Webcam capture recognized: {text_result}", 'info')
                else:
                    error_message = "No clear sign detected from webcam. Please ensure your hand is visible and performing a recognized sign."
                    flash(error_message, 'warning')
            
            # After processing a webcam capture, clear the live prediction so it doesn't get captured again
            with prediction_lock:
                shared_current_prediction = ""
                shared_prediction_confidence = 0.0

    return render_template('conversion.html',
                           text=text_result,
                           audio=audio_file_url,
                           error=error_message)

@app.route('/video_feed')
def video_feed():
    """Streams the webcam feed to the client."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    """Renders the About page."""
    return render_template('about.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Handles feedback submission."""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        remark = request.form.get('remark')
        
        # Log feedback (for production, save to database or send email)
        logger.info(f"Feedback received from {name} ({email}): {remark}")
        flash("Thank you for your feedback! We appreciate it.", 'success') # Use Flask's flash messages
        return redirect(url_for('feedback')) # Redirect after POST to prevent re-submission
        
    return render_template('feedback.html')

@app.route('/contact')
def contact():
    """Renders the Contact page."""
    return render_template('contact.html')

@app.route('/api/current_prediction')
def current_prediction_api():
    """API endpoint to get current webcam prediction for AJAX updates."""
    with prediction_lock: # Safely read shared variables
        return jsonify({
            'prediction': shared_current_prediction,
            'confidence': float(shared_prediction_confidence)
        })

# --- Application Startup ---
if __name__ == '__main__':
    # Initialize all necessary components ONCE when the script starts
    initialize_application_components()

    # Run the Flask app
    # debug=True is good for development, but should be False in production
    # host='0.0.0.0' makes the server accessible from other devices on your local network
    # port=5000 is a common port, ensure it's not in use
    app.run(debug=True, host='0.0.0.0', port=5000)