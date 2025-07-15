"""
Data Collection Module for Lane Keep Assist System
Records camera footage and steering angle data during human driving
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
import logging
import threading
import sys

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import DATA_COLLECTION, RAW_DATA_PATH, STEERING


class DataCollector:
    """
    Data collector for recording driving data
    Captures camera feed and steering angle measurements
    """
    
    def __init__(self, output_dir=None, camera_index=0):
        """
        Initialize data collector
        
        Args:
            output_dir: Directory to save collected data
            camera_index: Camera device index
        """
        self.output_dir = output_dir or RAW_DATA_PATH
        self.camera_index = camera_index
        self.is_recording = False
        self.frame_count = 0
        self.steering_data = []
        self.session_id = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize camera
        self.camera = None
        self.setup_camera()
        
        # Steering input (can be from joystick, keyboard, or serial)
        self.steering_angle = 0.0
        self.last_steering_update = time.time()
        
    def setup_logging(self):
        """Setup logging for data collection"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'data_collection.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_camera(self):
        """Initialize camera capture"""
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, DATA_COLLECTION['camera_width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DATA_COLLECTION['camera_height'])
        self.camera.set(cv2.CAP_PROP_FPS, DATA_COLLECTION['fps'])
        
        self.logger.info(f"Camera {self.camera_index} initialized successfully")
        
    def create_session(self, session_name=None):
        """
        Create a new data collection session
        
        Args:
            session_name: Name for the session
            
        Returns:
            str: Session ID
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_id = session_name
        self.session_dir = os.path.join(self.output_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(self.session_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, 'logs'), exist_ok=True)
        
        self.logger.info(f"Created session: {self.session_id}")
        return self.session_id
        
    def start_recording(self, session_name=None):
        """
        Start recording data
        
        Args:
            session_name: Name for the recording session
        """
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return
        
        # Create session
        self.create_session(session_name)
        
        # Reset counters
        self.frame_count = 0
        self.steering_data = []
        
        # Start recording
        self.is_recording = True
        self.logger.info("Started recording")
        
    def stop_recording(self):
        """Stop recording and save data"""
        if not self.is_recording:
            self.logger.warning("No recording in progress")
            return
        
        self.is_recording = False
        self.save_steering_data()
        self.save_session_metadata()
        
        self.logger.info(f"Recording stopped. Captured {self.frame_count} frames")
        
    def update_steering_angle(self, angle):
        """
        Update current steering angle
        
        Args:
            angle: Steering angle in degrees (-25 to +25)
        """
        # Clamp steering angle to valid range
        angle = max(STEERING['min_steering_angle'], 
                   min(STEERING['max_steering_angle'], angle))
        
        self.steering_angle = angle
        self.last_steering_update = time.time()
        
    def capture_frame(self):
        """
        Capture a single frame and record steering data
        
        Returns:
            tuple: (success, frame, steering_angle, timestamp)
        """
        if not self.camera.isOpened():
            return False, None, None, None
        
        success, frame = self.camera.read()
        if not success:
            return False, None, None, None
        
        timestamp = time.time()
        
        if self.is_recording:
            # Save frame
            frame_filename = f"{DATA_COLLECTION['image_prefix']}{self.frame_count:06d}.{DATA_COLLECTION['image_format']}"
            frame_path = os.path.join(self.session_dir, 'images', frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Record steering data
            self.steering_data.append({
                'frame_id': self.frame_count,
                'timestamp': timestamp,
                'steering_angle': self.steering_angle,
                'image_path': frame_path
            })
            
            self.frame_count += 1
            
        return True, frame, self.steering_angle, timestamp
        
    def save_steering_data(self):
        """Save steering angle data to CSV"""
        if not self.steering_data:
            self.logger.warning("No steering data to save")
            return
        
        df = pd.DataFrame(self.steering_data)
        csv_path = os.path.join(self.session_dir, DATA_COLLECTION['steering_log_file'])
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Saved steering data to {csv_path}")
        
    def save_session_metadata(self):
        """Save session metadata"""
        metadata = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'frame_count': self.frame_count,
            'duration': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            'camera_settings': {
                'width': DATA_COLLECTION['camera_width'],
                'height': DATA_COLLECTION['camera_height'],
                'fps': DATA_COLLECTION['fps']
            },
            'steering_range': {
                'min_angle': STEERING['min_steering_angle'],
                'max_angle': STEERING['max_steering_angle']
            }
        }
        
        metadata_path = os.path.join(self.session_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata to {metadata_path}")
        
    def run_collection_loop(self):
        """
        Main data collection loop
        Shows camera feed and records data
        """
        self.logger.info("Starting data collection loop. Press 'q' to quit, 'r' to start/stop recording")
        
        while True:
            success, frame, steering_angle, timestamp = self.capture_frame()
            
            if not success:
                self.logger.error("Failed to capture frame")
                break
            
            # Add overlay information
            overlay_frame = self.add_overlay(frame, steering_angle, timestamp)
            
            # Display frame
            cv2.imshow('Data Collection', overlay_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord('a'):  # Simulate left turn
                self.update_steering_angle(self.steering_angle - 5)
            elif key == ord('d'):  # Simulate right turn
                self.update_steering_angle(self.steering_angle + 5)
            elif key == ord('s'):  # Center steering
                self.update_steering_angle(0)
        
        # Cleanup
        if self.is_recording:
            self.stop_recording()
        
        self.cleanup()
        
    def add_overlay(self, frame, steering_angle, timestamp):
        """
        Add information overlay to the frame
        
        Args:
            frame: Input frame
            steering_angle: Current steering angle
            timestamp: Frame timestamp
            
        Returns:
            frame: Frame with overlay
        """
        overlay_frame = frame.copy()
        
        # Recording status
        status_text = "RECORDING" if self.is_recording else "STANDBY"
        status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        cv2.putText(overlay_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Steering angle
        cv2.putText(overlay_frame, f"Steering: {steering_angle:.1f}°", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Frame count
        if self.is_recording:
            cv2.putText(overlay_frame, f"Frame: {self.frame_count}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Steering visualization
        self.draw_steering_indicator(overlay_frame, steering_angle)
        
        return overlay_frame
        
    def draw_steering_indicator(self, frame, steering_angle):
        """
        Draw steering wheel indicator
        
        Args:
            frame: Frame to draw on
            steering_angle: Current steering angle
        """
        h, w = frame.shape[:2]
        center_x = w - 100
        center_y = h - 100
        radius = 60
        
        # Draw steering wheel circle
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Calculate steering indicator position
        angle_rad = np.radians(steering_angle)
        end_x = int(center_x + radius * 0.8 * np.sin(angle_rad))
        end_y = int(center_y - radius * 0.8 * np.cos(angle_rad))
        
        # Draw steering indicator
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
        cv2.circle(frame, (end_x, end_y), 5, (0, 255, 0), -1)
        
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info("Data collector cleaned up")


class JoystickDataCollector(DataCollector):
    """
    Data collector with joystick support for steering input
    """
    
    def __init__(self, output_dir=None, camera_index=0):
        super().__init__(output_dir, camera_index)
        self.setup_joystick()
        
    def setup_joystick(self):
        """Setup joystick for steering input"""
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() == 0:
                self.logger.warning("No joystick detected")
                self.joystick = None
                return
            
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.logger.info(f"Joystick connected: {self.joystick.get_name()}")
            
        except ImportError:
            self.logger.warning("Pygame not installed, joystick support disabled")
            self.joystick = None
            
    def read_joystick(self):
        """Read joystick input and update steering angle"""
        if not self.joystick:
            return
        
        try:
            import pygame
            pygame.event.pump()
            
            # Read joystick axis (typically axis 0 for steering)
            axis_value = self.joystick.get_axis(0)
            
            # Convert to steering angle
            steering_angle = axis_value * STEERING['max_steering_angle']
            self.update_steering_angle(steering_angle)
            
        except Exception as e:
            self.logger.error(f"Error reading joystick: {e}")


if __name__ == "__main__":
    # Test data collection
    collector = DataCollector()
    
    try:
        collector.run_collection_loop()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        collector.cleanup()