"""
Lightweight model architectures for stress detection on Jetson Orin Nano.
Includes GRU-based and TCN-based models optimized for temporal pose data.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Optional


class StressDetectionModel:
    """
    Factory class for creating lightweight stress detection models.
    """
    
    @staticmethod
    def build_gru_model(input_shape: Tuple[int, int], 
                       dropout_rate: float = 0.3,
                       recurrent_dropout: float = 0.2) -> Model:
        """
        Build a GRU-based model for stress detection.
        
        Args:
            input_shape: (sequence_length, num_features)
            dropout_rate: Dropout rate for regularization
            recurrent_dropout: Recurrent dropout rate for GRU layers
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # First GRU layer with return sequences
            layers.GRU(64, 
                      return_sequences=True, 
                      dropout=dropout_rate,
                      recurrent_dropout=recurrent_dropout,
                      name='gru_1'),
            layers.BatchNormalization(),
            
            # Second GRU layer
            layers.GRU(32, 
                      dropout=dropout_rate,
                      recurrent_dropout=recurrent_dropout,
                      name='gru_2'),
            layers.BatchNormalization(),
            
            # Dense layers for final prediction
            layers.Dense(16, activation='relu', name='dense_1'),
            layers.Dropout(dropout_rate),
            layers.Dense(8, activation='relu', name='dense_2'),
            layers.Dense(1, activation='sigmoid', name='stress_output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        return model
    
    @staticmethod
    def build_tcn_model(input_shape: Tuple[int, int],
                       num_filters: int = 32,
                       kernel_size: int = 3,
                       num_blocks: int = 3,
                       dropout_rate: float = 0.3) -> Model:
        """
        Build a Temporal Convolutional Network (TCN) for stress detection.
        
        Args:
            input_shape: (sequence_length, num_features)
            num_filters: Number of filters in conv layers
            kernel_size: Kernel size for convolutions
            num_blocks: Number of TCN blocks
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Initial convolution
        x = layers.Conv1D(num_filters, 1, padding='same')(x)
        
        # TCN blocks with increasing dilation
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            
            # First conv in residual block
            conv1 = layers.Conv1D(num_filters, kernel_size, 
                                 dilation_rate=dilation_rate, 
                                 padding='same')(x)
            conv1 = layers.BatchNormalization()(conv1)
            conv1 = layers.Activation('relu')(conv1)
            conv1 = layers.Dropout(dropout_rate)(conv1)
            
            # Second conv in residual block
            conv2 = layers.Conv1D(num_filters, kernel_size,
                                 dilation_rate=dilation_rate,
                                 padding='same')(conv1)
            conv2 = layers.BatchNormalization()(conv2)
            conv2 = layers.Activation('relu')(conv2)
            conv2 = layers.Dropout(dropout_rate)(conv2)
            
            # Residual connection
            if x.shape[-1] != num_filters:
                x = layers.Conv1D(num_filters, 1, padding='same')(x)
            x = layers.Add()([x, conv2])
        
        # Global pooling and final layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid', name='stress_output')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        return model
    
    @staticmethod
    def build_lightweight_lstm(input_shape: Tuple[int, int],
                              dropout_rate: float = 0.3) -> Model:
        """
        Build a lightweight LSTM model as an alternative to GRU.
        
        Args:
            input_shape: (sequence_length, num_features)
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # LSTM layers
            layers.LSTM(32, return_sequences=True, dropout=dropout_rate, name='lstm_1'),
            layers.BatchNormalization(),
            layers.LSTM(16, dropout=dropout_rate, name='lstm_2'),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(8, activation='relu', name='dense_1'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid', name='stress_output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        return model
    
    @staticmethod
    def build_hybrid_model(input_shape: Tuple[int, int],
                          dropout_rate: float = 0.3) -> Model:
        """
        Build a hybrid model combining CNN and GRU for temporal and spatial features.
        
        Args:
            input_shape: (sequence_length, num_features)
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=input_shape)
        
        # 1D CNN for local temporal patterns
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # GRU for long-term temporal dependencies
        x = layers.GRU(32, return_sequences=True, dropout=dropout_rate)(x)
        x = layers.GRU(16, dropout=dropout_rate)(x)
        x = layers.BatchNormalization()(x)
        
        # Final prediction layers
        x = layers.Dense(8, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='stress_output')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        return model


class ModelUtils:
    """Utility functions for model training and evaluation."""
    
    @staticmethod
    def get_callbacks(model_save_path: str, 
                     patience: int = 10,
                     reduce_lr_patience: int = 5) -> list:
        """
        Get standard callbacks for training.
        
        Args:
            model_save_path: Path to save best model
            patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callbacks
    
    @staticmethod
    def count_parameters(model: Model) -> int:
        """Count trainable parameters in the model."""
        return model.count_params()
    
    @staticmethod
    def model_summary(model: Model) -> None:
        """Print detailed model summary."""
        print("Model Architecture:")
        model.summary()
        print(f"\nTotal parameters: {ModelUtils.count_parameters(model):,}")
        
        # Estimate memory usage (rough approximation)
        total_params = model.count_params()
        memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        print(f"Estimated memory usage: {memory_mb:.2f} MB")
    
    @staticmethod
    def convert_to_tflite(model: Model, 
                         save_path: str,
                         quantize: bool = True) -> None:
        """
        Convert Keras model to TensorFlow Lite for deployment.
        
        Args:
            model: Trained Keras model
            save_path: Path to save .tflite file
            quantize: Whether to apply quantization
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable Select TF ops and disable TensorList lowering to fix conversion errors
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # For better quantization, you can add representative dataset
            # converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to: {save_path}")
        
        # Check file size
        import os
        file_size = os.path.getsize(save_path) / (1024 * 1024)
        print(f"TFLite model size: {file_size:.2f} MB")


# Example usage and model comparison
if __name__ == "__main__":
    # Example input shape: 30 frames, 50+ features per frame
    input_shape = (30, 52)  # sequence_length=30, num_features=52
    
    print("Building different model architectures...")
    
    # GRU model (recommended for start)
    print("\n1. GRU Model:")
    gru_model = StressDetectionModel.build_gru_model(input_shape)
    ModelUtils.model_summary(gru_model)
    
    # TCN model
    print("\n2. TCN Model:")
    tcn_model = StressDetectionModel.build_tcn_model(input_shape)
    ModelUtils.model_summary(tcn_model)
    
    # Lightweight LSTM
    print("\n3. Lightweight LSTM:")
    lstm_model = StressDetectionModel.build_lightweight_lstm(input_shape)
    ModelUtils.model_summary(lstm_model)
    
    # Hybrid model
    print("\n4. Hybrid CNN-GRU Model:")
    hybrid_model = StressDetectionModel.build_hybrid_model(input_shape)
    ModelUtils.model_summary(hybrid_model)
    
    print("\nModel comparison complete!")
    print("Recommendation: Start with GRU model for best balance of performance and efficiency.")
