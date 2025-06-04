import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleASLPreprocessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,  # Kaggle dataset has single hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def extract_hand_landmarks(self, image_path):
        """Extract hand landmarks from image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                
                return np.array(landmarks)
            else:
                # Return zeros if no hand detected
                return np.zeros(63)  # 21 landmarks * 3 coordinates
                
        except Exception as e:
            logger.warning(f"Error processing {image_path}: {e}")
            return None
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position"""
        if landmarks is None or len(landmarks) == 0:
            return np.zeros(63)
        
        # Reshape to (21, 3) for easier manipulation
        reshaped = landmarks.reshape(21, 3)
        
        # Use wrist (landmark 0) as reference point
        wrist = reshaped[0]
        
        # Normalize relative to wrist
        normalized = reshaped - wrist
        
        # Flatten back to 1D array
        return normalized.flatten()

class KaggleASLDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.preprocessor = KaggleASLPreprocessor()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, limit_per_class=None):
        """Load and preprocess the Kaggle ASL Alphabet dataset"""
        X, y = [], []
        
        # Path to training data
        train_path = os.path.join(self.dataset_path, 'asl_alphabet_train')
        
        # Get all class directories
        classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        logger.info(f"Found {len(classes)} classes: {classes}")
        
        for class_name in classes:
            class_path = os.path.join(train_path, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples per class if specified
            if limit_per_class:
                image_files = image_files[:limit_per_class]
            
            logger.info(f"Processing class '{class_name}': {len(image_files)} images")
            
            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                image_path = os.path.join(class_path, image_file)
                
                # Extract landmarks
                landmarks = self.preprocessor.extract_hand_landmarks(image_path)
                
                if landmarks is not None:
                    # Normalize landmarks
                    normalized_landmarks = self.preprocessor.normalize_landmarks(landmarks)
                    
                    X.append(normalized_landmarks)
                    y.append(class_name)
        
        return np.array(X), np.array(y)
    
    def prepare_sequences(self, X, sequence_length=10):
        """Convert static landmarks to sequences by data augmentation"""
        X_sequences = []
        
        for landmarks in X:
            # Create sequence by adding small random variations
            sequence = []
            for i in range(sequence_length):
                # Add small gaussian noise for variation
                noise = np.random.normal(0, 0.01, landmarks.shape)
                augmented_landmarks = landmarks + noise
                sequence.append(augmented_landmarks)
            
            X_sequences.append(sequence)
        
        return np.array(X_sequences)

class ASLClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build LSTM model for ASL classification"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=10, 
                restore_best_weights=True, 
                monitor='val_accuracy'
            ),
            ModelCheckpoint(
                'best_asl_model.h5', 
                save_best_only=True, 
                monitor='val_accuracy',
                save_weights_only=False
            ),
            ReduceLROnPlateau(
                factor=0.5, 
                patience=5, 
                min_lr=1e-7,
                monitor='val_loss'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test, class_names):
        """Evaluate model and generate reports"""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        report = classification_report(y_test, y_pred_classes, target_names=class_names)
        logger.info("Classification Report:")
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, cm
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuration
    DATASET_PATH = "asl_alphabet_dataset"  # Update this path
    SEQUENCE_LENGTH = 10
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    LIMIT_PER_CLASS = 1000  # Set to None to use all data
    
    # Create output directory
    os.makedirs('model_outputs', exist_ok=True)
    
    logger.info("Starting ASL Alphabet training pipeline...")
    
    # Load dataset
    logger.info("Loading and preprocessing dataset...")
    loader = KaggleASLDataLoader(DATASET_PATH)
    X, y = loader.load_data(limit_per_class=LIMIT_PER_CLASS)
    
    logger.info(f"Loaded {len(X)} samples")
    logger.info(f"Feature shape: {X[0].shape}")
    logger.info(f"Classes: {sorted(set(y))}")
    
    # Encode labels
    y_encoded = loader.label_encoder.fit_transform(y)
    class_names = loader.label_encoder.classes_
    
    # Save label encoder
    with open('model_outputs/label_encoder.pkl', 'wb') as f:
        pickle.dump(loader.label_encoder, f)
    
    # Create sequences for LSTM
    logger.info("Creating sequence data...")
    X_sequences = loader.prepare_sequences(X, SEQUENCE_LENGTH)
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_sequences, y_encoded, test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), 
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Create and build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, features)
    num_classes = len(class_names)
    
    classifier = ASLClassifier(input_shape, num_classes)
    model = classifier.build_model()
    
    logger.info("Model architecture:")
    model.summary()
    
    # Train model
    logger.info("Starting training...")
    history = classifier.train(
        X_train, y_train, X_val, y_val, 
        epochs=50, batch_size=32
    )
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Load best model
    model.load_weights('best_asl_model.h5')
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate detailed evaluation
    report, cm = classifier.evaluate_model(X_test, y_test, class_names)
    
    # Save final model
    model.save('model_outputs/final_asl_model.h5')
    
    # Save model configuration
    config = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'sequence_length': SEQUENCE_LENGTH,
        'class_names': class_names.tolist(),
        'test_accuracy': float(test_accuracy)
    }
    
    with open('model_outputs/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final test accuracy: {test_accuracy:.4f}")
    logger.info("Model saved in 'model_outputs/' directory")

if __name__ == "__main__":
    main()