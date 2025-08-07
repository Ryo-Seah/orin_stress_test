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
        
        self.data_loader = BOLDDataset(
            bold_root=self.config['bold_root'],
            sequence_length=self.config['sequence_length'],
            min_confidence=self.config['min_confidence']
        )
    
    
    def plot_vad_distribution(self, y, split_name="train"):
        """Plot the distribution of VAD values."""
        import matplotlib.pyplot as plt
        import numpy as np
        vad_names = ['Valence', 'Arousal', 'Dominance']
        bin_edges = np.arange(0, 10.5, 0.5)  # 0.0 to 10.0 in 0.5 steps
        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for i in range(3):
            counts, _ = np.histogram(y[:, i], bins=bin_edges)
            axes[i].bar(bin_labels, counts, width=0.8, align='center', alpha=0.7)
            axes[i].set_xticks(range(len(bin_labels)))
            axes[i].set_xticklabels(bin_labels, rotation=90, fontsize=8)
            axes[i].set_xlabel(f"{vad_names[i]} Bin")
            axes[i].set_title(f"{vad_names[i]} Distribution\n(Total: {int(np.sum(counts))})")
            for idx, count in enumerate(counts):
                axes[i].text(idx, count, str(count), ha='center', va='bottom', fontsize=7)
        axes[0].set_ylabel("Count")
        plt.suptitle(f"VAD Distribution in {split_name} set (bin size=0.05)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', f'vad_distribution_{split_name}.png'))
        plt.close()

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        print("Loading and preprocessing data...")
        
        # Load raw data
        X_train, y_train = self.data_loader.load_dataset('train')
        X_val, y_val = self.data_loader.load_dataset('val')
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("No valid data found! Check dataset paths and format.")
        
        # Plot VAD distributions
        self.plot_vad_distribution(y_train, split_name="train")
        self.plot_vad_distribution(y_val, split_name="val")
        
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
        

        self.data_info = {
            'train_samples':   len(X_train),
            'val_samples':     len(X_val),
            'sequence_length': X_train.shape[1],
            'num_features':    X_train.shape[2],
        }
        
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
        y_pred = self.model.predict(self.val_dataset, verbose=0)
        y_true = []
        for _, labels in self.val_dataset:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)
        
        # Calculate additional metrics for each VAD component
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
        self.evaluation_results = {
            'val_loss': float(val_loss),
            'val_mae': float(val_mae),
            'val_mse': float(val_mse),
            'val_rmse': float(np.sqrt(val_mse)),
            'r2_score': r2.tolist(),
            'mae': mae.tolist()
        };
        
        print(f"Evaluation Results (per VAD component):")
        for i, comp in enumerate(['Valence', 'Arousal', 'Dominance']):
            print(f"  {comp}: R²={r2[i]:.4f}, MAE={mae[i]:.4f}")
        
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
        """Plot prediction analysis for VAD."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        vad_names = ['Valence', 'Arousal', 'Dominance']
        for i in range(3):
            # Scatter plot: True vs Predicted
            axes[0, i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
            axes[0, i].plot([y_true[:, i].min(), y_true[:, i].max()], [y_true[:, i].min(), y_true[:, i].max()], 'r--', lw=2)
            axes[0, i].set_xlabel(f'True {vad_names[i]}')
            axes[0, i].set_ylabel(f'Predicted {vad_names[i]}')
            axes[0, i].set_title(f'True vs Predicted {vad_names[i]}')
            
            # Residuals
            residuals = y_true[:, i] - y_pred[:, i]
            axes[1, i].hist(residuals, bins=30, alpha=0.7)
            axes[1, i].set_xlabel(f'{vad_names[i]} Prediction Error')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'{vad_names[i]} Error Distribution')
        
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
    
    parser.add_argument('--model_type', type=str, default='gru', 
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
