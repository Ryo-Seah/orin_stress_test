"""
Main training script for stress detection using BOLD dataset.
Handles the complete training pipeline from data loading to model evaluation.
"""

import os
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from data_processing import BOLDDataset
from models import StressDetectionModel, ModelUtils
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class StressDetectionTrainer:
    """Main trainer class for stress detection models."""
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.model = None
        self.data_loader = None
        self.history = None
      
    def setup_data_loader(self):
        """Initialize and configure the data loader."""
        print("Setting up data loader...")
        print(f"🔍 DEBUG: BOLD root path from config: {self.config['bold_root']}")
        print(f"🔍 DEBUG: Does path exist? {os.path.exists(self.config['bold_root'])}")
        
        # List what's actually in the directory
        if os.path.exists(self.config['bold_root']):
            print(f"🔍 DEBUG: Contents of {self.config['bold_root']}:")
            for item in os.listdir(self.config['bold_root']):
                print(f"  - {item}")
        
        self.data_loader = BOLDDataset(
            bold_root=self.config['bold_root'],
            sequence_length=self.config['sequence_length'],
            overlap_ratio=self.config['overlap_ratio'],
            min_confidence=self.config['min_confidence']
        )
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        print("Loading and preprocessing data...")
        
        # Load raw data
        X_train, y_train = self.data_loader.load_dataset('train')
        X_val, y_val = self.data_loader.load_dataset('val')
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("No valid data found! Check dataset paths and format.")
        
        # Fit scaler on training data and transform both sets
        print("Fitting feature scaler...")
        self.data_loader.fit_scaler(X_train)
        X_train_scaled = self.data_loader.transform_features(X_train)
        X_val_scaled = self.data_loader.transform_features(X_val)
        
        # Create TensorFlow datasets
        self.train_dataset = self.data_loader.create_tf_dataset(
            X_train_scaled, y_train, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        self.val_dataset = self.data_loader.create_tf_dataset(
            X_val_scaled, y_val, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Store data info
        self.data_info = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'sequence_length': X_train.shape[1],
            'num_features': X_train.shape[2],
            'train_stress_mean': float(np.mean(y_train)),
            'train_stress_std': float(np.std(y_train)),
            'val_stress_mean': float(np.mean(y_val)),
            'val_stress_std': float(np.std(y_val))
        }
        
        print(f"Data loading completed:")
        print(f"  Training samples: {self.data_info['train_samples']}")
        print(f"  Validation samples: {self.data_info['val_samples']}")
        print(f"  Input shape: ({self.data_info['sequence_length']}, {self.data_info['num_features']})")
        print(f"  Train stress range: {y_train.min():.3f} - {y_train.max():.3f}")
        print(f"  Val stress range: {y_val.min():.3f} - {y_val.max():.3f}")
        
    def build_model(self):
        """Build the specified model architecture."""
        print(f"Building {self.config['model_type']} model...")
        
        input_shape = (self.data_info['sequence_length'], self.data_info['num_features'])
        
        if self.config['model_type'] == 'gru':
            self.model = StressDetectionModel.build_gru_model(
                input_shape, 
                dropout_rate=self.config['dropout_rate']
            )
        elif self.config['model_type'] == 'tcn':
            self.model = StressDetectionModel.build_tcn_model(
                input_shape,
                dropout_rate=self.config['dropout_rate']
            )
        elif self.config['model_type'] == 'lstm':
            self.model = StressDetectionModel.build_lightweight_lstm(
                input_shape,
                dropout_rate=self.config['dropout_rate']
            )
        elif self.config['model_type'] == 'hybrid':
            self.model = StressDetectionModel.build_hybrid_model(
                input_shape,
                dropout_rate=self.config['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
        
        ModelUtils.model_summary(self.model)
        
    def train_model(self):
        """Train the model."""
        print("Starting model training...")
        
        # Prepare callbacks
        model_save_path = os.path.join(
            self.config['output_dir'], 
            'models', 
            f"best_{self.config['model_type']}_model.keras"  # Use new Keras format
        )
        
        callbacks = ModelUtils.get_callbacks(
            model_save_path=model_save_path,
            patience=self.config['early_stopping_patience'],
            reduce_lr_patience=self.config['reduce_lr_patience']
        )
        
        # Train the model
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("Evaluating model...")
        
        # Evaluate on validation set
        val_loss, val_mae, val_mse = self.model.evaluate(self.val_dataset, verbose=0)
        
        # Generate predictions for analysis
        y_pred = self.model.predict(self.val_dataset, verbose=0).flatten()
        
        # Get true values (need to reconstruct from dataset)
        y_true = []
        for _, labels in self.val_dataset:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)
        
        # Calculate additional metrics
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        self.evaluation_results = {
            'val_loss': float(val_loss),
            'val_mae': float(val_mae),
            'val_mse': float(val_mse),
            'val_rmse': float(np.sqrt(val_mse)),
            'r2_score': float(r2),
            'mape': float(mape)
        }
        
        print(f"Evaluation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  RMSE: {np.sqrt(val_mse):.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.4f}")
        
        return y_true, y_pred
        
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'training_history.png'))
        plt.close()
        
    def plot_predictions(self, y_true, y_pred):
        """Plot prediction analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Scatter plot: True vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Stress Score')
        axes[0, 0].set_ylabel('Predicted Stress Score')
        axes[0, 0].set_title('True vs Predicted Stress Scores')
        
        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Stress Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Distribution of predictions
        axes[1, 0].hist(y_true, alpha=0.7, label='True', bins=30)
        axes[1, 0].hist(y_pred, alpha=0.7, label='Predicted', bins=30)
        axes[1, 0].set_xlabel('Stress Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Stress Scores')
        axes[1, 0].legend()
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Prediction Errors')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'prediction_analysis.png'))
        plt.close()
        
    def save_results(self):
        """Save training results and configuration."""
        results = {
            'config': self.config,
            'data_info': self.data_info,
            'evaluation_results': self.evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.config['output_dir'], 'logs', 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
    def convert_to_tflite(self):
        """Convert trained model to TensorFlow Lite."""
        if self.model is None:
            return
            
        tflite_path = os.path.join(
            self.config['output_dir'], 
            'models', 
            f"{self.config['model_type']}_stress_model.tflite"
        )
        
        ModelUtils.convert_to_tflite(
            self.model, 
            tflite_path, 
            quantize=True
        )
        
    def run_training(self):
        """Run the complete training pipeline."""
        print("Starting stress detection training pipeline...")
        print("=" * 50)
        
        try:
            # 1. Setup
            self.setup_data_loader()
            
            # 2. Data loading and preprocessing
            self.load_and_preprocess_data()
            
            # 3. Model building
            self.build_model()
            
            # 4. Training
            self.train_model()
            
            # 5. Evaluation
            y_true, y_pred = self.evaluate_model()
            
            # 6. Visualization
            self.plot_training_history()
            self.plot_predictions(y_true, y_pred)
            
            # 7. Save results
            self.save_results()
            
            # 8. Convert to TFLite for deployment
            self.convert_to_tflite()
            
            print("=" * 50)
            print("Training pipeline completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()


def get_default_config():
    """Get default training configuration."""
    return {
        # Data parameters
        'bold_root': '/Users/RyoSeah/Desktop/Stress_Detection/BOLD_public',
        'sequence_length': 30,
        'overlap_ratio': 0.5,
        'min_confidence': 0.3,
        
        # Model parameters
        'model_type': 'gru',  # 'gru', 'tcn', 'lstm', 'hybrid'
        'dropout_rate': 0.3,
        
        # Training parameters
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        
        # Output
        'output_dir': '/Users/RyoSeah/Desktop/Stress_Detection/training_results'
    }


def main():
    """Main function to run training with command line arguments."""
    parser = argparse.ArgumentParser(description='Train stress detection model on BOLD dataset')
    
    # Add the missing bold_root argument
    parser.add_argument('--bold_root', type=str, 
                       default='/Users/RyoSeah/Desktop/Stress_Detection/BOLD_public',
                       help='Path to BOLD dataset root directory')
    
    parser.add_argument('--model_type', type=str, default='tcn', 
                       choices=['gru', 'tcn', 'lstm', 'hybrid'],
                       help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--output_dir', type=str, 
                       default='/Users/RyoSeah/Desktop/Stress_Detection/training_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Get default config and update with command line arguments
    config = get_default_config()
    config.update(vars(args))
    
    # Create trainer and run
    trainer = StressDetectionTrainer(config)
    trainer.run_training()


if __name__ == "__main__":
    main()
