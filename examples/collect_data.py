#!/usr/bin/env python3
"""
Example script to collect training data for Lane Keep Assist
This script demonstrates how to record camera footage and steering angles
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.data_collector import DataCollector, JoystickDataCollector
from config.config import RAW_DATA_PATH


def main():
    """Main data collection function"""
    parser = argparse.ArgumentParser(description='Collect training data for Lane Keep Assist')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for collected data')
    parser.add_argument('--session_name', type=str, default=None,
                       help='Name for the data collection session')
    parser.add_argument('--camera_index', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--use_joystick', action='store_true',
                       help='Use joystick for steering input')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_collection.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Create data collector
        if args.use_joystick:
            logger.info("Using joystick data collector")
            collector = JoystickDataCollector(
                output_dir=args.output_dir or RAW_DATA_PATH,
                camera_index=args.camera_index
            )
        else:
            logger.info("Using keyboard data collector")
            collector = DataCollector(
                output_dir=args.output_dir or RAW_DATA_PATH,
                camera_index=args.camera_index
            )
        
        logger.info("Data collection setup complete")
        logger.info("Controls:")
        logger.info("  - Press 'r' to start/stop recording")
        logger.info("  - Press 'q' to quit")
        logger.info("  - Press 'a' to steer left (keyboard mode)")
        logger.info("  - Press 'd' to steer right (keyboard mode)")
        logger.info("  - Press 's' to center steering (keyboard mode)")
        
        # Start collection loop
        collector.run_collection_loop()
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise
    finally:
        logger.info("Data collection completed")


if __name__ == "__main__":
    main()