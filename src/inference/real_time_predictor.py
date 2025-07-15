"""
Real-time Inference Module for Lane Keep Assist System
Handles real-time steering angle prediction during vehicle operation
"""

import cv2
import numpy as np
import os
import sys
import time
import logging
import threading
from collections import deque
import json

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import INFERENCE, STEERING, MODEL_PATH, SYSTEM
from src.model.pilotnet import PilotNet
from src.preprocessing.image_processor import ImageProcessor


class RealTimePredictor:
    """
    Real-time predictor for steering angle from camera input
    """
    
    def __init__(self, model_path=None, camera_index=0):
        """
        Initialize real-time predictor
        
        Args:
            model_path: Path to trained PilotNet model
            camera_index: Camera device index
        """
        self.model_path = model_path or os.path.join(MODEL_PATH, INFERENCE['model_file'])
        self.camera_index = camera_index
        
        # Initialize components
        self.model = None
        self.processor = ImageProcessor()
        self.camera = None
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        self.last_prediction = 0.0
        
        # Performance monitoring
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = None
        
        # Threading
        self.is_running = False
        self.prediction_thread = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize system
        self.initialize_system()
        
    def setup_logging(self):
        """Setup logging for inference"""
        logging.basicConfig(
            level=getattr(logging, SYSTEM['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('inference.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_system(self):
        """Initialize the inference system"""
        self.logger.info("Initializing real-time predictor...")
        
        # Load model
        self.load_model()
        
        # Initialize camera
        self.initialize_camera()
        
        self.logger.info("Real-time predictor initialized successfully")
        
    def load_model(self):
        """Load the trained PilotNet model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.logger.info(f"Loading model from {self.model_path}")
        
        self.model = PilotNet()
        self.model.load_model(self.model_path)
        
        self.logger.info("Model loaded successfully")
        
    def initialize_camera(self):
        """Initialize camera capture"""
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, INFERENCE['fps_target'])
        
        self.logger.info(f"Camera {self.camera_index} initialized")
        
    def predict_steering(self, image):
        """
        Predict steering angle from image
        
        Args:
            image: Input image
            
        Returns:
            float: Predicted steering angle
        """
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.processor.preprocess_image(image, augment=False)
        
        # Make prediction
        prediction = self.model.predict_steering(processed_image)
        
        # Apply steering multiplier
        prediction = prediction * INFERENCE['steering_multiplier']
        
        # Clamp to valid range
        prediction = np.clip(prediction, 
                           STEERING['min_steering_angle'], 
                           STEERING['max_steering_angle'])
        
        # Apply smoothing
        smoothed_prediction = self.smooth_prediction(prediction)
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        return smoothed_prediction
        
    def smooth_prediction(self, prediction):
        """
        Apply smoothing to prediction
        
        Args:
            prediction: Raw prediction
            
        Returns:
            float: Smoothed prediction
        """
        # Add to buffer
        self.prediction_buffer.append(prediction)
        
        # Apply moving average
        if len(self.prediction_buffer) > 0:
            smoothed = np.mean(self.prediction_buffer)
        else:
            smoothed = prediction
        
        # Apply additional smoothing factor
        smoothed = (STEERING['smoothing_factor'] * self.last_prediction + 
                   (1 - STEERING['smoothing_factor']) * smoothed)
        
        # Limit maximum change per frame
        max_change = INFERENCE['max_steering_change']
        if abs(smoothed - self.last_prediction) > max_change:
            if smoothed > self.last_prediction:
                smoothed = self.last_prediction + max_change
            else:
                smoothed = self.last_prediction - max_change
        
        self.last_prediction = smoothed
        return smoothed
        
    def capture_and_predict(self):
        """
        Capture frame and predict steering angle
        
        Returns:
            tuple: (success, frame, steering_angle, inference_time)
        """
        success, frame = self.camera.read()
        
        if not success:
            return False, None, None, None
        
        start_time = time.time()
        steering_angle = self.predict_steering(frame)
        inference_time = time.time() - start_time
        
        return True, frame, steering_angle, inference_time
        
    def add_overlay(self, frame, steering_angle, inference_time):
        """
        Add information overlay to frame
        
        Args:
            frame: Input frame
            steering_angle: Predicted steering angle
            inference_time: Inference time
            
        Returns:
            frame: Frame with overlay
        """
        overlay_frame = frame.copy()
        
        # Current prediction
        cv2.putText(overlay_frame, f"Steering: {steering_angle:.2f}°", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Inference time
        cv2.putText(overlay_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        if self.frame_count > 0 and self.start_time:
            fps = self.frame_count / (time.time() - self.start_time)
            cv2.putText(overlay_frame, f"FPS: {fps:.1f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Average inference time
        if self.frame_count > 0:
            avg_inference = self.total_inference_time / self.frame_count
            cv2.putText(overlay_frame, f"Avg Inference: {avg_inference*1000:.1f}ms", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Steering visualization
        self.draw_steering_visualization(overlay_frame, steering_angle)
        
        return overlay_frame
        
    def draw_steering_visualization(self, frame, steering_angle):
        """
        Draw steering visualization on frame
        
        Args:
            frame: Frame to draw on
            steering_angle: Steering angle to visualize
        """
        h, w = frame.shape[:2]
        
        # Steering bar
        bar_x = w - 50
        bar_y = 50
        bar_height = 200
        bar_width = 20
        
        # Draw bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Draw center line
        center_y = bar_y + bar_height // 2
        cv2.line(frame, (bar_x, center_y), (bar_x + bar_width, center_y),
                (255, 255, 255), 2)
        
        # Draw steering indicator
        normalized_angle = steering_angle / STEERING['max_steering_angle']
        indicator_y = int(center_y - normalized_angle * bar_height // 2)
        indicator_y = max(bar_y, min(bar_y + bar_height, indicator_y))
        
        cv2.rectangle(frame, (bar_x, indicator_y - 5), (bar_x + bar_width, indicator_y + 5),
                     (0, 255, 0), -1)
        
        # Steering wheel visualization
        wheel_center_x = w - 100
        wheel_center_y = h - 100
        wheel_radius = 50
        
        # Draw wheel
        cv2.circle(frame, (wheel_center_x, wheel_center_y), wheel_radius, (255, 255, 255), 2)
        
        # Draw steering indicator
        angle_rad = np.radians(steering_angle)
        end_x = int(wheel_center_x + wheel_radius * 0.8 * np.sin(angle_rad))
        end_y = int(wheel_center_y - wheel_radius * 0.8 * np.cos(angle_rad))
        
        cv2.line(frame, (wheel_center_x, wheel_center_y), (end_x, end_y), (0, 255, 0), 3)
        cv2.circle(frame, (end_x, end_y), 5, (0, 255, 0), -1)
        
    def run_inference_loop(self):
        """Run the main inference loop"""
        self.logger.info("Starting inference loop...")
        self.start_time = time.time()
        self.is_running = True
        
        try:
            while self.is_running:
                success, frame, steering_angle, inference_time = self.capture_and_predict()
                
                if not success:
                    self.logger.error("Failed to capture frame")
                    break
                
                # Add overlay if visualization is enabled
                if INFERENCE['visualization']:
                    display_frame = self.add_overlay(frame, steering_angle, inference_time)
                    cv2.imshow('Lane Keep Assist - Real-time Inference', display_frame)
                
                # Save predictions if enabled
                if INFERENCE['save_predictions']:
                    self.save_prediction(steering_angle, inference_time)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_metrics()
                elif key == ord('s'):
                    self.save_performance_stats()
                
                # Control frame rate
                time.sleep(max(0, (1.0 / INFERENCE['fps_target']) - inference_time))
                
        except KeyboardInterrupt:
            self.logger.info("Inference interrupted by user")
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
        finally:
            self.stop_inference()
            
    def save_prediction(self, steering_angle, inference_time):
        """
        Save prediction to file
        
        Args:
            steering_angle: Predicted steering angle
            inference_time: Inference time
        """
        prediction_data = {
            'timestamp': time.time(),
            'frame_count': self.frame_count,
            'steering_angle': float(steering_angle),
            'inference_time': float(inference_time)
        }
        
        # Append to predictions log
        with open('predictions.jsonl', 'a') as f:
            f.write(json.dumps(prediction_data) + '\n')
            
    def reset_metrics(self):
        """Reset performance metrics"""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        self.logger.info("Performance metrics reset")
        
    def save_performance_stats(self):
        """Save performance statistics"""
        if self.frame_count > 0:
            total_time = time.time() - self.start_time
            avg_inference = self.total_inference_time / self.frame_count
            fps = self.frame_count / total_time
            
            stats = {
                'timestamp': time.time(),
                'total_frames': self.frame_count,
                'total_time': total_time,
                'average_inference_time': avg_inference,
                'fps': fps,
                'model_path': self.model_path
            }
            
            with open('performance_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"Performance stats saved - FPS: {fps:.1f}, "
                           f"Avg inference: {avg_inference*1000:.1f}ms")
            
    def stop_inference(self):
        """Stop inference and cleanup"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # Save final performance stats
        self.save_performance_stats()
        
        self.logger.info("Inference stopped and cleaned up")
        
    def get_current_prediction(self):
        """
        Get current steering prediction (for external use)
        
        Returns:
            float: Current steering angle prediction
        """
        return self.last_prediction
        
    def is_model_loaded(self):
        """
        Check if model is loaded
        
        Returns:
            bool: True if model is loaded
        """
        return self.model is not None
        
    def get_performance_stats(self):
        """
        Get current performance statistics
        
        Returns:
            dict: Performance statistics
        """
        if self.frame_count > 0 and self.start_time:
            total_time = time.time() - self.start_time
            return {
                'frames_processed': self.frame_count,
                'total_time': total_time,
                'fps': self.frame_count / total_time,
                'average_inference_time': self.total_inference_time / self.frame_count,
                'current_prediction': self.last_prediction
            }
        return {}


class LaneKeepAssistSystem:
    """
    Complete Lane Keep Assist System
    Integrates real-time prediction with vehicle control
    """
    
    def __init__(self, model_path=None, camera_index=0):
        """
        Initialize Lane Keep Assist System
        
        Args:
            model_path: Path to trained model
            camera_index: Camera device index
        """
        self.predictor = RealTimePredictor(model_path, camera_index)
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.lka_enabled = False
        self.manual_override = False
        
    def enable_lka(self):
        """Enable Lane Keep Assist"""
        self.lka_enabled = True
        self.logger.info("Lane Keep Assist enabled")
        
    def disable_lka(self):
        """Disable Lane Keep Assist"""
        self.lka_enabled = False
        self.logger.info("Lane Keep Assist disabled")
        
    def set_manual_override(self, override):
        """
        Set manual override state
        
        Args:
            override: True to enable manual override
        """
        self.manual_override = override
        self.logger.info(f"Manual override: {override}")
        
    def get_steering_command(self):
        """
        Get steering command for vehicle
        
        Returns:
            float: Steering command (0 if LKA disabled or manual override)
        """
        if not self.lka_enabled or self.manual_override:
            return 0.0
        
        return self.predictor.get_current_prediction()
        
    def run_system(self):
        """Run the complete Lane Keep Assist system"""
        self.logger.info("Starting Lane Keep Assist System...")
        
        # Enable LKA by default
        self.enable_lka()
        
        # Start inference loop
        self.predictor.run_inference_loop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Lane Keep Assist real-time inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained PilotNet model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Override config if needed
    if args.no_visualization:
        INFERENCE['visualization'] = False
    
    try:
        # Create and run predictor
        predictor = RealTimePredictor(args.model, args.camera)
        predictor.run_inference_loop()
        
    except Exception as e:
        print(f"Error: {e}")
        raise