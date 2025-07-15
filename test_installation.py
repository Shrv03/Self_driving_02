#!/usr/bin/env python3
"""
Test script to verify Lane Keep Assist system installation and basic functionality
"""

import sys
import os
import numpy as np
import traceback

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
        return False
    
    return True

def test_project_structure():
    """Test that the project structure is correct"""
    print("\nTesting project structure...")
    
    required_dirs = [
        'src',
        'src/model',
        'src/data_collection',
        'src/preprocessing',
        'src/training',
        'src/inference',
        'src/utils',
        'config',
        'examples',
        'data',
        'data/raw',
        'data/processed',
        'data/models'
    ]
    
    required_files = [
        'config/config.py',
        'src/model/pilotnet.py',
        'src/data_collection/data_collector.py',
        'src/preprocessing/image_processor.py',
        'src/training/trainer.py',
        'src/inference/real_time_predictor.py',
        'src/utils/visualization.py',
        'examples/collect_data.py',
        'examples/train_model.py',
        'examples/run_inference.py'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/")
            all_exist = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            all_exist = False
    
    return all_exist

def test_config_loading():
    """Test that configuration can be loaded"""
    print("\nTesting configuration loading...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from config.config import MODEL, TRAINING, STEERING, INFERENCE
        
        print("✓ Configuration loaded successfully")
        print(f"  Model input shape: {MODEL['input_shape']}")
        print(f"  Training batch size: {TRAINING['batch_size']}")
        print(f"  Steering range: {STEERING['min_steering_angle']}° to {STEERING['max_steering_angle']}°")
        print(f"  Inference FPS target: {INFERENCE['fps_target']}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test that PilotNet model can be created"""
    print("\nTesting PilotNet model creation...")
    
    try:
        from src.model.pilotnet import PilotNet
        
        # Create model
        model = PilotNet()
        model.build_model()
        
        print("✓ PilotNet model created successfully")
        print(f"  Model parameters: {model.model.count_params():,}")
        print(f"  Input shape: {model.model.input_shape}")
        print(f"  Output shape: {model.model.output_shape}")
        
        # Test prediction with dummy data
        dummy_input = np.random.rand(1, 66, 200, 3) * 255
        prediction = model.predict_steering(dummy_input)
        print(f"  Test prediction: {prediction:.3f}°")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_image_processing():
    """Test that image processing works"""
    print("\nTesting image processing...")
    
    try:
        from src.preprocessing.image_processor import ImageProcessor
        
        # Create processor
        processor = ImageProcessor()
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = processor.preprocess_image(dummy_image, augment=False)
        print(f"✓ Image preprocessing successful")
        print(f"  Input shape: {dummy_image.shape}")
        print(f"  Output shape: {processed.shape}")
        
        # Test augmentation
        augmented = processor.preprocess_image(dummy_image, augment=True)
        print(f"✓ Image augmentation successful")
        print(f"  Augmented shape: {augmented.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        traceback.print_exc()
        return False

def test_camera_availability():
    """Test camera availability"""
    print("\nTesting camera availability...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera available (index 0)")
                print(f"  Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print("✗ Camera found but cannot capture frames")
                cap.release()
                return False
        else:
            print("✗ No camera found at index 0")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability for training"""
    print("\nTesting GPU availability...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) available:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True
        else:
            print("⚠ No GPU detected (CPU training will be slower)")
            return True
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Lane Keep Assist System Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Configuration Loading", test_config_loading),
        ("Model Creation", test_model_creation),
        ("Image Processing", test_image_processing),
        ("Camera Availability", test_camera_availability),
        ("GPU Availability", test_gpu_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before using the system.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)