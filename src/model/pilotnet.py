"""
PilotNet CNN Architecture for Lane Keep Assist
Based on Nvidia's End-to-End Deep Learning for Self-Driving Cars
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
import os

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import MODEL, TRAINING, SYSTEM


class PilotNet:
    """
    PilotNet CNN model for behavioral cloning in autonomous driving
    Architecture:
    - 5 Convolutional layers with normalization
    - 3 Fully connected layers
    - Dropout for regularization
    - Single output for steering angle prediction
    """
    
    def __init__(self, input_shape=None, learning_rate=None):
        """
        Initialize PilotNet model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape or MODEL['input_shape']
        self.learning_rate = learning_rate or TRAINING['learning_rate']
        self.model = None
        
    def build_model(self):
        """
        Build the PilotNet CNN architecture
        
        Returns:
            tf.keras.Model: Compiled PilotNet model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_images')
        
        # Normalization layer (pixel values from 0-255 to -1 to 1)
        x = layers.Lambda(lambda x: (x / 127.5) - 1.0, name='normalization')(inputs)
        
        # Convolutional layers
        conv_configs = MODEL['conv_layers']
        
        # First convolutional layer
        x = layers.Conv2D(
            filters=conv_configs[0]['filters'],
            kernel_size=conv_configs[0]['kernel_size'],
            strides=conv_configs[0]['strides'],
            activation=conv_configs[0]['activation'],
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='conv2d_1'
        )(x)
        
        # Second convolutional layer
        x = layers.Conv2D(
            filters=conv_configs[1]['filters'],
            kernel_size=conv_configs[1]['kernel_size'],
            strides=conv_configs[1]['strides'],
            activation=conv_configs[1]['activation'],
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='conv2d_2'
        )(x)
        
        # Third convolutional layer
        x = layers.Conv2D(
            filters=conv_configs[2]['filters'],
            kernel_size=conv_configs[2]['kernel_size'],
            strides=conv_configs[2]['strides'],
            activation=conv_configs[2]['activation'],
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='conv2d_3'
        )(x)
        
        # Fourth convolutional layer
        x = layers.Conv2D(
            filters=conv_configs[3]['filters'],
            kernel_size=conv_configs[3]['kernel_size'],
            strides=conv_configs[3]['strides'],
            activation=conv_configs[3]['activation'],
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='conv2d_4'
        )(x)
        
        # Fifth convolutional layer
        x = layers.Conv2D(
            filters=conv_configs[4]['filters'],
            kernel_size=conv_configs[4]['kernel_size'],
            strides=conv_configs[4]['strides'],
            activation=conv_configs[4]['activation'],
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='conv2d_5'
        )(x)
        
        # Flatten layer
        x = layers.Flatten(name='flatten')(x)
        
        # Fully connected layers
        dense_configs = MODEL['dense_layers']
        
        # First fully connected layer
        x = layers.Dense(
            dense_configs[0],
            activation='relu',
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='dense_1'
        )(x)
        x = layers.Dropout(MODEL['dropout_rate'], name='dropout_1')(x)
        
        # Second fully connected layer
        x = layers.Dense(
            dense_configs[1],
            activation='relu',
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='dense_2'
        )(x)
        x = layers.Dropout(MODEL['dropout_rate'], name='dropout_2')(x)
        
        # Third fully connected layer
        x = layers.Dense(
            dense_configs[2],
            activation='relu',
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='dense_3'
        )(x)
        x = layers.Dropout(MODEL['dropout_rate'], name='dropout_3')(x)
        
        # Output layer (steering angle prediction)
        outputs = layers.Dense(
            dense_configs[3],
            activation='linear',
            kernel_regularizer=regularizers.l2(MODEL['l2_regularization']),
            name='steering_output'
        )(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='PilotNet')
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=TRAINING['loss'],
            metrics=TRAINING['metrics']
        )
        
        return self.model
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_steering(self, image):
        """
        Predict steering angle from input image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            float: Predicted steering angle
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Ensure image has correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)
        return float(prediction[0][0])
    
    def get_layer_outputs(self, image, layer_names=None):
        """
        Get intermediate layer outputs for visualization
        
        Args:
            image: Input image
            layer_names: List of layer names to get outputs from
            
        Returns:
            dict: Dictionary of layer outputs
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        if layer_names is None:
            layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5']
        
        # Create a model that outputs intermediate layers
        outputs = [self.model.get_layer(name).output for name in layer_names]
        intermediate_model = models.Model(inputs=self.model.input, outputs=outputs)
        
        # Get predictions
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        intermediate_outputs = intermediate_model.predict(image, verbose=0)
        
        # Return as dictionary
        return {name: output for name, output in zip(layer_names, intermediate_outputs)}


def create_pilotnet_model(input_shape=None, learning_rate=None):
    """
    Factory function to create and build PilotNet model
    
    Args:
        input_shape: Input image shape
        learning_rate: Learning rate for optimizer
        
    Returns:
        PilotNet: Built PilotNet model instance
    """
    pilotnet = PilotNet(input_shape, learning_rate)
    pilotnet.build_model()
    return pilotnet


if __name__ == "__main__":
    # Test the model creation
    print("Creating PilotNet model...")
    model = create_pilotnet_model()
    
    print("\nModel Summary:")
    model.get_model_summary()
    
    print(f"\nModel input shape: {model.model.input_shape}")
    print(f"Model output shape: {model.model.output_shape}")
    
    # Test prediction with random input
    test_image = np.random.rand(1, 66, 200, 3) * 255
    prediction = model.predict_steering(test_image)
    print(f"\nTest prediction: {prediction}")