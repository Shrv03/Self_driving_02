# Lane Keep Assist System with PilotNet

A comprehensive Lane Keep Assist (LKA) system for Advanced Driver Assistance Systems (ADAS) using deep learning and behavior cloning. This project implements Nvidia's PilotNet CNN architecture to learn steering behavior from human driving data.

## Overview

This system enables a vehicle to maintain lane position using only front camera input without requiring LIDAR, GPS, or HD maps. The approach uses end-to-end deep learning where the model learns to predict steering angles directly from camera images by mimicking human driving behavior.

### Key Features

- **Behavior Cloning**: Learns steering behavior from human driving demonstrations
- **PilotNet Architecture**: Implements Nvidia's proven CNN architecture
- **Real-time Inference**: Provides steering predictions at 30 FPS
- **Vision-only Approach**: No dependency on LIDAR, GPS, or HD maps
- **Comprehensive Training Pipeline**: Complete data collection, preprocessing, and training workflow
- **Performance Monitoring**: Real-time visualization and performance metrics
- **Modular Design**: Easy to extend and customize

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Collection│    │   Preprocessing │    │     Training    │
│                 │    │                 │    │                 │
│ • Camera Input  │───▶│ • Image Resize  │───▶│ • PilotNet CNN  │
│ • Steering Data │    │ • Normalization │    │ • Behavior Clone│
│ • Synchronized  │    │ • Augmentation  │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Real-time LKA  │    │    Inference    │    │  Trained Model  │
│                 │    │                 │    │                 │
│ • Lane Keeping  │◀───│ • Steering Pred │◀───│ • PilotNet.h5   │
│ • Smoothing     │    │ • Real-time     │    │ • Weights       │
│ • Safety Limits │    │ • Visualization │    │ • Architecture  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- Camera (USB webcam or vehicle camera)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd lane-keep-assist
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## Quick Start

### 1. Data Collection

Collect training data by recording camera footage and steering angles:

```bash
python examples/collect_data.py --output_dir data/my_training_session
```

**Controls:**
- `r`: Start/stop recording
- `q`: Quit
- `a`: Steer left (keyboard mode)
- `d`: Steer right (keyboard mode)
- `s`: Center steering

### 2. Model Training

Train the PilotNet model with your collected data:

```bash
python examples/train_model.py --data_dir data/my_training_session --model_name my_pilotnet
```

**Training will:**
- Preprocess images (resize, crop, normalize)
- Apply data augmentation
- Train PilotNet CNN architecture
- Generate performance visualizations
- Save trained model

### 3. Real-time Inference

Run lane keep assist with your trained model:

```bash
python examples/run_inference.py --model data/models/my_pilotnet_best.h5
```

**Controls:**
- `q`: Quit
- `r`: Reset performance metrics
- `s`: Save performance statistics

## Project Structure

```
lane-keep-assist/
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── data_collection/
│   │   └── data_collector.py  # Data recording utilities
│   ├── preprocessing/
│   │   └── image_processor.py # Image preprocessing & augmentation
│   ├── model/
│   │   └── pilotnet.py        # PilotNet CNN architecture
│   ├── training/
│   │   └── trainer.py         # Training pipeline
│   ├── inference/
│   │   └── real_time_predictor.py # Real-time inference
│   └── utils/
│       └── visualization.py   # Visualization utilities
├── examples/
│   ├── collect_data.py        # Data collection example
│   ├── train_model.py         # Training example
│   └── run_inference.py       # Inference example
├── data/
│   ├── raw/                   # Raw collected data
│   ├── processed/             # Processed datasets
│   └── models/                # Trained models
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## PilotNet Architecture

The system implements Nvidia's PilotNet CNN architecture optimized for steering prediction:

```
Input Image (66x200x3)
        ↓
Normalization Layer
        ↓
Conv2D(24, 5x5, stride=2) → ReLU
        ↓
Conv2D(36, 5x5, stride=2) → ReLU
        ↓
Conv2D(48, 5x5, stride=2) → ReLU
        ↓
Conv2D(64, 3x3, stride=1) → ReLU
        ↓
Conv2D(64, 3x3, stride=1) → ReLU
        ↓
Flatten
        ↓
Dense(100) → ReLU → Dropout(0.5)
        ↓
Dense(50) → ReLU → Dropout(0.5)
        ↓
Dense(10) → ReLU → Dropout(0.5)
        ↓
Dense(1) → Steering Angle
```

## Configuration

The system is highly configurable through `config/config.py`:

### Key Settings

```python
# Model Architecture
MODEL = {
    'input_shape': (66, 200, 3),
    'conv_layers': [...],
    'dense_layers': [100, 50, 10, 1],
    'dropout_rate': 0.5,
    'l2_regularization': 0.001
}

# Training Parameters
TRAINING = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 10
}

# Steering Limits
STEERING = {
    'max_steering_angle': 25.0,  # degrees
    'min_steering_angle': -25.0,
    'smoothing_factor': 0.8
}
```

## Usage Examples

### Advanced Data Collection

```bash
# Use joystick input
python examples/collect_data.py --use_joystick --session_name highway_driving

# Specify camera and output directory
python examples/collect_data.py --camera_index 1 --output_dir /path/to/data
```

### Advanced Training

```bash
# Train with custom parameters
python examples/train_model.py \
    --data_dir data/highway_session \
    --model_name highway_pilotnet \
    --epochs 150 \
    --batch_size 64 \
    --no_balance

# Train with specific output directory
python examples/train_model.py \
    --data_dir data/mixed_conditions \
    --output_dir models/production \
    --model_name production_v1
```

### Advanced Inference

```bash
# Run without visualization (headless)
python examples/run_inference.py \
    --model models/production_v1_best.h5 \
    --no_visualization \
    --save_predictions

# Run full Lane Keep Assist system
python examples/run_inference.py \
    --model models/production_v1_best.h5 \
    --full_system
```

## Performance Monitoring

The system provides comprehensive performance monitoring:

### Training Metrics
- Training/validation loss curves
- Mean Absolute Error (MAE)
- R² score
- Steering angle distribution analysis

### Real-time Metrics
- Inference time (ms)
- Frames per second (FPS)
- Steering angle predictions
- Performance statistics

### Visualization
- Model activation maps
- Filter weight visualization
- Prediction analysis plots
- Performance dashboard

## Data Collection Best Practices

1. **Diverse Conditions**: Collect data in various lighting, weather, and road conditions
2. **Quality Driving**: Ensure smooth, consistent steering behavior
3. **Balanced Dataset**: Include equal amounts of straight, left, and right turns
4. **Sufficient Data**: Collect at least 30 minutes of driving data
5. **Multiple Sessions**: Record separate sessions for different scenarios

## Model Training Tips

1. **Data Preprocessing**: Ensure consistent image preprocessing
2. **Data Augmentation**: Use appropriate augmentation to improve generalization
3. **Early Stopping**: Monitor validation loss to prevent overfitting
4. **Learning Rate**: Use learning rate scheduling for better convergence
5. **Regularization**: Apply dropout and L2 regularization

## Safety Considerations

⚠️ **Important Safety Notes:**

1. **Testing Environment**: Only test in safe, controlled environments
2. **Human Supervision**: Always have a human driver ready to take control
3. **Speed Limits**: Test at low speeds initially
4. **Emergency Override**: Implement manual override capabilities
5. **Fail-safe Mechanisms**: Include steering angle limits and safety checks

## Troubleshooting

### Common Issues

1. **Camera Not Found**
   ```bash
   # Check available cameras
   python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
   ```

2. **GPU Memory Issues**
   ```bash
   # Reduce batch size in config.py
   TRAINING['batch_size'] = 16
   ```

3. **Poor Model Performance**
   - Collect more diverse training data
   - Increase training epochs
   - Adjust learning rate
   - Check data preprocessing

4. **Slow Inference**
   - Optimize model architecture
   - Use GPU acceleration
   - Reduce image resolution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- Nvidia for the PilotNet architecture
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools

## References

1. Bojarski, M., et al. "End to End Learning for Self-Driving Cars." arXiv:1604.07316 (2016).
2. Nvidia PilotNet: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
3. Udacity Self-Driving Car Nanodegree: https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the configuration in `config/config.py`

---

**⚠️ Disclaimer**: This system is for educational and research purposes only. Do not use in production vehicles without proper safety validation and testing.