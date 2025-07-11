"""
Evaluation script for trained stress detection models.
Includes inference testing and performance analysis.
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import BOLDDataLoader
import tensorflow as tf


class StressDetectionEvaluator:
    """Evaluator for trained stress detection models."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to trained model (.h5 or .tflite)
            config_path: Path to training configuration JSON
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.is_tflite = model_path.endswith('.tflite')
        
        # Load configuration if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                results = json.load(f)
                self.config = results.get('config', {})
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from: {self.model_path}")
        
        if self.is_tflite:
            # Load TensorFlow Lite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("TensorFlow Lite model loaded successfully")
        else:
            # Load Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            print("Keras model loaded successfully")
    
    def predict_keras(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Keras model."""
        return self.model.predict(X, verbose=0)
    
    def predict_tflite(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using TensorFlow Lite model."""
        predictions = []
        
        for i in range(len(X)):
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], X[i:i+1].astype(np.float32))
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output[0])
        
        return np.array(predictions)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the appropriate method."""
        if self.is_tflite:
            return self.predict_tflite(X)
        else:
            return self.predict_keras(X)
    
    def benchmark_inference_speed(self, X_sample: np.ndarray, num_runs: int = 100):
        """Benchmark inference speed."""
        print(f"Benchmarking inference speed with {num_runs} runs...")
        
        # Warm up
        for _ in range(10):
            _ = self.predict(X_sample[:1])
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.predict(X_sample[:1])
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Inference FPS: {fps:.1f}")
        
        return avg_time, fps
    
    def evaluate_on_test_set(self):
        """Evaluate model on test dataset if available."""
        if not self.config:
            print("No configuration available. Cannot load test data.")
            return None
        
        # Setup data loader with saved configuration
        data_loader = BOLDDataLoader(
            bold_root=self.config.get('bold_root', '/Users/RyoSeah/Downloads/BOLD_public'),
            sequence_length=self.config.get('sequence_length', 30),
            overlap_ratio=self.config.get('overlap_ratio', 0.5),
            min_confidence=self.config.get('min_confidence', 0.3)
        )
        
        # Try to load test data (might not exist)
        try:
            X_test, y_test = data_loader.load_dataset('test')
            if len(X_test) == 0:
                print("No test data available. Using validation data.")
                X_test, y_test = data_loader.load_dataset('val')
        except:
            print("Loading validation data for evaluation...")
            X_test, y_test = data_loader.load_dataset('val')
        
        if len(X_test) == 0:
            print("No evaluation data available.")
            return None
        
        # Load the scaler (this is a limitation - ideally scaler should be saved)
        print("Note: Recomputing scaler on available data. For production, save the scaler separately.")
        X_train, _ = data_loader.load_dataset('train')
        data_loader.fit_scaler(X_train)
        X_test_scaled = data_loader.transform_features(X_test)
        
        # Make predictions
        print(f"Making predictions on {len(X_test)} samples...")
        y_pred = self.predict(X_test_scaled).flatten()
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'num_samples': len(X_test),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mean_true': float(np.mean(y_test)),
            'mean_pred': float(np.mean(y_pred)),
            'std_true': float(np.std(y_test)),
            'std_pred': float(np.std(y_pred))
        }
        
        print("Evaluation Results:")
        print(f"  Samples: {results['num_samples']}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  R²: {results['r2_score']:.4f}")
        print(f"  Mean True: {results['mean_true']:.4f}")
        print(f"  Mean Pred: {results['mean_pred']:.4f}")
        
        # Benchmark inference speed
        avg_time, fps = self.benchmark_inference_speed(X_test_scaled[:10])
        results['avg_inference_time_ms'] = avg_time * 1000
        results['inference_fps'] = fps
        
        return results
    
    def predict_single_sequence(self, pose_sequence: np.ndarray, 
                               data_loader: BOLDDataLoader = None) -> float:
        """
        Predict stress score for a single pose sequence.
        
        Args:
            pose_sequence: Array of shape (sequence_length, 18, 3) - raw pose data
            data_loader: Data loader for feature extraction (optional)
            
        Returns:
            Predicted stress score (0-1)
        """
        if data_loader is None and self.config:
            data_loader = BOLDDataLoader(
                bold_root=self.config.get('bold_root', ''),
                sequence_length=self.config.get('sequence_length', 30),
                overlap_ratio=self.config.get('overlap_ratio', 0.5),
                min_confidence=self.config.get('min_confidence', 0.3)
            )
        
        if data_loader is None:
            raise ValueError("Need data_loader for feature extraction")
        
        # Extract features
        features = data_loader.extract_pose_features(pose_sequence)
        velocities = data_loader.compute_velocity_features(features)
        padded_velocities = np.vstack([np.zeros((1, velocities.shape[1])), velocities])
        combined_features = np.hstack([features, padded_velocities])
        
        # Normalize features (note: this requires the scaler to be fitted)
        # In production, you should save and load the scaler
        features_scaled = combined_features.reshape(1, *combined_features.shape)
        
        # Make prediction
        prediction = self.predict(features_scaled)
        
        return float(prediction[0])


def test_model_on_sample_data():
    """Test the evaluator with sample data."""
    # Example usage
    model_path = "/Users/RyoSeah/Desktop/Stress_Detection/training_results/models/best_gru_model.h5"
    config_path = "/Users/RyoSeah/Desktop/Stress_Detection/training_results/logs/training_results.json"
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train a model first using train.py")
        return
    
    try:
        evaluator = StressDetectionEvaluator(model_path, config_path)
        results = evaluator.evaluate_on_test_set()
        
        if results:
            print("\nEvaluation completed successfully!")
            
            # Save evaluation results
            output_dir = os.path.dirname(os.path.dirname(model_path))
            eval_results_path = os.path.join(output_dir, 'logs', 'evaluation_results.json')
            
            results['timestamp'] = datetime.now().isoformat()
            results['model_path'] = model_path
            
            with open(eval_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {eval_results_path}")
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        config_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        evaluator = StressDetectionEvaluator(model_path, config_path)
        evaluator.evaluate_on_test_set()
    else:
        test_model_on_sample_data()
