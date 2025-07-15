#!/usr/bin/env python3
"""
Example script to run real-time inference with trained PilotNet model
This script demonstrates how to use the trained model for lane keep assist
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.real_time_predictor import RealTimePredictor, LaneKeepAssistSystem
from config.config import MODEL_PATH, INFERENCE


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Run real-time inference for Lane Keep Assist')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained PilotNet model (.h5 file)')
    parser.add_argument('--camera_index', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Disable visualization window')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--full_system', action='store_true',
                       help='Run full Lane Keep Assist system')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('inference.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    try:
        if args.full_system:
            logger.info("Starting full Lane Keep Assist system...")
            system = LaneKeepAssistSystem(
                model_path=args.model,
                camera_index=args.camera_index
            )
            
            logger.info("Lane Keep Assist system controls:")
            logger.info("  - Press 'q' to quit")
            logger.info("  - Press 'e' to enable/disable LKA")
            logger.info("  - Press 'm' to toggle manual override")
            logger.info("  - Press 'r' to reset performance metrics")
            logger.info("  - Press 's' to save performance statistics")
            
            system.run_system()
            
        else:
            logger.info("Starting real-time inference...")
            logger.info(f"Model: {args.model}")
            logger.info(f"Camera: {args.camera_index}")
            
            # Override config if needed
            if args.no_visualization:
                from config.config import INFERENCE
                INFERENCE['visualization'] = False
                
            if args.save_predictions:
                INFERENCE['save_predictions'] = True
            
            # Create predictor
            predictor = RealTimePredictor(
                model_path=args.model,
                camera_index=args.camera_index
            )
            
            logger.info("Real-time inference controls:")
            logger.info("  - Press 'q' to quit")
            logger.info("  - Press 'r' to reset performance metrics")
            logger.info("  - Press 's' to save performance statistics")
            
            # Run inference
            predictor.run_inference_loop()
            
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
    finally:
        logger.info("Inference completed")


if __name__ == "__main__":
    main()