#!/usr/bin/env python3
"""
Example script to train PilotNet model for Lane Keep Assist
This script demonstrates the complete training pipeline
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.trainer import PilotNetTrainer
from src.preprocessing.image_processor import ImageProcessor
from src.utils.visualization import create_performance_dashboard
from config.config import TRAINING, MODEL_PATH


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train PilotNet model for Lane Keep Assist')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data (with images/ and steering_log.csv)')
    parser.add_argument('--model_name', type=str, default='pilotnet_lka',
                       help='Name for the trained model')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size (default from config)')
    parser.add_argument('--no_balance', action='store_true',
                       help='Don\'t balance the dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for models and results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Create trainer
        trainer = PilotNetTrainer(model_save_path=args.output_dir)
        
        logger.info(f"Starting training pipeline for {args.model_name}")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Balance dataset: {not args.no_balance}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        train_data, val_data, test_data = trainer.load_and_preprocess_data(
            args.data_dir, 
            balance_data=not args.no_balance
        )
        
        # Create model
        logger.info("Creating PilotNet model...")
        trainer.create_model()
        
        # Train model
        logger.info("Starting model training...")
        history = trainer.train_model(
            train_data, 
            val_data, 
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        metrics, predictions = trainer.evaluate_model(test_data)
        
        # Create visualizations
        logger.info("Creating performance visualizations...")
        
        # Training history plot
        history_plot_path = os.path.join(trainer.model_save_path, f"{args.model_name}_training_history.png")
        trainer.plot_training_history(history_plot_path)
        
        # Prediction analysis plot
        predictions_plot_path = os.path.join(trainer.model_save_path, f"{args.model_name}_predictions.png")
        trainer.plot_predictions(test_data, predictions, predictions_plot_path)
        
        # Performance dashboard
        dashboard_path = os.path.join(trainer.model_save_path, f"{args.model_name}_dashboard.png")
        create_performance_dashboard(trainer.model, test_data, predictions, history.history, dashboard_path)
        
        # Print summary
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {trainer.model_save_path}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        logger.info(f"Test performance:")
        logger.info(f"  - R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  - MAE: {metrics['mae']:.4f}°")
        logger.info(f"  - RMSE: {metrics['rmse']:.4f}°")
        
        # Save training summary
        summary_path = os.path.join(trainer.model_save_path, f"{args.model_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Lane Keep Assist Training Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Data Directory: {args.data_dir}\n")
            f.write(f"Training Samples: {len(train_data[0])}\n")
            f.write(f"Validation Samples: {len(val_data[0])}\n")
            f.write(f"Test Samples: {len(test_data[0])}\n\n")
            f.write(f"Training Configuration:\n")
            f.write(f"  - Epochs: {len(history.history['loss'])}\n")
            f.write(f"  - Batch Size: {args.batch_size or TRAINING['batch_size']}\n")
            f.write(f"  - Learning Rate: {TRAINING['learning_rate']}\n")
            f.write(f"  - Balanced Dataset: {not args.no_balance}\n\n")
            f.write(f"Final Results:\n")
            f.write(f"  - Training Loss: {history.history['loss'][-1]:.4f}\n")
            f.write(f"  - Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
            f.write(f"  - Test R² Score: {metrics['r2_score']:.4f}\n")
            f.write(f"  - Test MAE: {metrics['mae']:.4f}°\n")
            f.write(f"  - Test RMSE: {metrics['rmse']:.4f}°\n")
            f.write(f"  - Max Error: {metrics['max_error']:.4f}°\n")
        
        logger.info(f"Training summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()