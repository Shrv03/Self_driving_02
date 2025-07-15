"""
Configuration file for Lane Keep Assist System
Contains all hyperparameters and settings for data collection, training, and inference
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(DATA_ROOT, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_ROOT, 'processed')
MODEL_PATH = os.path.join(DATA_ROOT, 'models')

# Data Collection Settings
DATA_COLLECTION = {
    'camera_width': 640,
    'camera_height': 480,
    'fps': 30,
    'video_format': 'mp4',
    'steering_log_file': 'steering_log.csv',
    'image_prefix': 'frame_',
    'image_format': 'jpg'
}

# Image Preprocessing Settings
IMAGE_PREPROCESSING = {
    'target_width': 200,
    'target_height': 66,
    'crop_top': 60,    # Remove sky and distant objects
    'crop_bottom': 25,  # Remove car hood
    'normalize': True,
    'yuv_conversion': True,  # PilotNet uses YUV color space
    'gaussian_blur': True,
    'blur_kernel_size': 3
}

# Data Augmentation Settings
DATA_AUGMENTATION = {
    'brightness_range': 0.2,
    'rotation_range': 5,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': False,  # Don't flip for driving
    'zoom_range': 0.1,
    'shadow_probability': 0.5,
    'brightness_probability': 0.5
}

# Model Architecture Settings (PilotNet)
MODEL = {
    'input_shape': (66, 200, 3),
    'conv_layers': [
        {'filters': 24, 'kernel_size': (5, 5), 'strides': (2, 2), 'activation': 'relu'},
        {'filters': 36, 'kernel_size': (5, 5), 'strides': (2, 2), 'activation': 'relu'},
        {'filters': 48, 'kernel_size': (5, 5), 'strides': (2, 2), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'activation': 'relu'}
    ],
    'dense_layers': [100, 50, 10, 1],
    'dropout_rate': 0.5,
    'l2_regularization': 0.001
}

# Training Settings
TRAINING = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'shuffle': True,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'model_checkpoint_period': 5,
    'tensorboard_log_dir': os.path.join(PROJECT_ROOT, 'logs')
}

# Steering Settings
STEERING = {
    'max_steering_angle': 25.0,  # degrees
    'min_steering_angle': -25.0,  # degrees
    'correction_factor': 0.2,    # for left/right camera correction
    'measurement_threshold': 0.1,  # minimum steering change to record
    'smoothing_factor': 0.8      # for real-time steering smoothing
}

# Real-time Inference Settings
INFERENCE = {
    'camera_index': 0,
    'model_file': 'pilotnet_model.h5',
    'prediction_threshold': 0.1,
    'fps_target': 30,
    'steering_multiplier': 1.0,
    'max_steering_change': 5.0,  # degrees per frame
    'visualization': True,
    'save_predictions': False
}

# System Settings
SYSTEM = {
    'log_level': 'INFO',
    'log_file': 'lka_system.log',
    'use_gpu': True,
    'gpu_memory_limit': 4096,  # MB
    'random_seed': 42
}

# Evaluation Settings
EVALUATION = {
    'test_split': 0.1,
    'metrics': ['mse', 'mae', 'r2_score'],
    'visualization_samples': 10,
    'confusion_matrix_bins': 50
}

# Dataset Settings
DATASET = {
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'min_samples_per_angle': 10,
    'balance_dataset': True,
    'remove_zero_steering': False,
    'steering_bins': 25
}

# Hardware Settings
HARDWARE = {
    'raspberry_pi': {
        'gpio_steering_pin': 18,
        'gpio_enable_pin': 19,
        'pwm_frequency': 50,
        'camera_resolution': (640, 480)
    },
    'arduino': {
        'serial_port': '/dev/ttyACM0',
        'baud_rate': 9600,
        'timeout': 1
    }
}

# Create directories if they don't exist
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, TRAINING['tensorboard_log_dir']]:
    os.makedirs(path, exist_ok=True)