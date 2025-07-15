"""
Training Module for Lane Keep Assist System
Handles model training, validation, and evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
import logging

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import TRAINING, MODEL_PATH, EVALUATION, SYSTEM
from src.model.pilotnet import PilotNet
from src.preprocessing.image_processor import ImageProcessor, batch_generator


class PilotNetTrainer:
    """
    Trainer class for PilotNet model
    """
    
    def __init__(self, model_save_path=None, log_dir=None):
        """
        Initialize trainer
        
        Args:
            model_save_path: Path to save trained model
            log_dir: Directory for training logs
        """
        self.model_save_path = model_save_path or MODEL_PATH
        self.log_dir = log_dir or TRAINING['tensorboard_log_dir']
        
        # Create directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.processor = ImageProcessor()
        self.training_history = None
        
        # Setup logging
        self.setup_logging()
        
        # Setup GPU
        self.setup_gpu()
        
    def setup_logging(self):
        """Setup logging for training"""
        logging.basicConfig(
            level=getattr(logging, SYSTEM['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gpu(self):
        """Setup GPU configuration"""
        if SYSTEM['use_gpu']:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    if SYSTEM['gpu_memory_limit']:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpus[0],
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=SYSTEM['gpu_memory_limit'])]
                        )
                    
                    self.logger.info(f"GPU configured: {len(gpus)} GPU(s) available")
                except RuntimeError as e:
                    self.logger.error(f"GPU configuration error: {e}")
            else:
                self.logger.warning("No GPU detected, using CPU")
        else:
            self.logger.info("Using CPU for training")
            
    def load_and_preprocess_data(self, data_dir, balance_data=True):
        """
        Load and preprocess training data
        
        Args:
            data_dir: Directory containing raw data
            balance_data: Whether to balance the dataset
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        self.logger.info(f"Loading data from {data_dir}")
        
        # Process dataset
        images, steering_angles = self.processor.process_dataset(
            data_dir, augment_data=True
        )
        
        self.logger.info(f"Loaded {len(images)} samples")
        
        # Balance dataset if requested
        if balance_data:
            self.logger.info("Balancing dataset...")
            images, steering_angles = self.processor.balance_dataset(images, steering_angles)
            self.logger.info(f"Balanced dataset: {len(images)} samples")
        
        # Split dataset
        train_data, val_data, test_data = self.processor.split_dataset(
            images, steering_angles
        )
        
        self.logger.info(f"Dataset split - Train: {len(train_data[0])}, "
                        f"Val: {len(val_data[0])}, Test: {len(test_data[0])}")
        
        return train_data, val_data, test_data
        
    def create_model(self):
        """
        Create and compile PilotNet model
        
        Returns:
            PilotNet: Compiled model
        """
        self.logger.info("Creating PilotNet model...")
        
        self.model = PilotNet()
        self.model.build_model()
        
        self.logger.info("Model created successfully")
        self.logger.info(f"Model parameters: {self.model.model.count_params()}")
        
        return self.model
        
    def setup_callbacks(self, model_name="pilotnet"):
        """
        Setup training callbacks
        
        Args:
            model_name: Name for model checkpoints
            
        Returns:
            list: List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.model_save_path, f"{model_name}_best.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=TRAINING['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAINING['reduce_lr_factor'],
            patience=TRAINING['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard_dir = os.path.join(self.log_dir, f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        self.logger.info(f"Callbacks configured: {len(callbacks)} callbacks")
        
        return callbacks
        
    def train_model(self, train_data, val_data, model_name="pilotnet", epochs=None, batch_size=None):
        """
        Train the PilotNet model
        
        Args:
            train_data: Training data (images, steering_angles)
            val_data: Validation data (images, steering_angles)
            model_name: Name for the model
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            History: Training history
        """
        if self.model is None:
            self.create_model()
        
        epochs = epochs or TRAINING['epochs']
        batch_size = batch_size or TRAINING['batch_size']
        
        self.logger.info(f"Starting training - Epochs: {epochs}, Batch size: {batch_size}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks(model_name)
        
        # Prepare data generators
        train_generator = batch_generator(
            train_data[0], train_data[1], 
            batch_size=batch_size, 
            augment=True
        )
        
        val_generator = batch_generator(
            val_data[0], val_data[1], 
            batch_size=batch_size, 
            augment=False
        )
        
        # Calculate steps per epoch
        train_steps = len(train_data[0]) // batch_size
        val_steps = len(val_data[0]) // batch_size
        
        # Train model
        self.training_history = self.model.model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Training completed")
        
        # Save final model
        final_model_path = os.path.join(self.model_save_path, f"{model_name}_final.h5")
        self.model.save_model(final_model_path)
        
        # Save training history
        self.save_training_history(model_name)
        
        return self.training_history
        
    def save_training_history(self, model_name):
        """
        Save training history to file
        
        Args:
            model_name: Name of the model
        """
        if self.training_history is None:
            self.logger.warning("No training history to save")
            return
        
        history_path = os.path.join(self.model_save_path, f"{model_name}_history.json")
        
        # Convert history to serializable format
        history_dict = {}
        for key, values in self.training_history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
        
    def evaluate_model(self, test_data, model_path=None):
        """
        Evaluate model performance
        
        Args:
            test_data: Test data (images, steering_angles)
            model_path: Path to saved model (if None, uses current model)
            
        Returns:
            dict: Evaluation metrics
        """
        if model_path:
            self.logger.info(f"Loading model from {model_path}")
            self.model = PilotNet()
            self.model.load_model(model_path)
        
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        self.logger.info("Evaluating model...")
        
        test_images, test_steering = test_data
        
        # Preprocess test images
        processed_test_images = []
        for img in test_images:
            processed_img = self.processor.preprocess_image(img, augment=False)
            processed_test_images.append(processed_img)
        
        processed_test_images = np.array(processed_test_images)
        
        # Make predictions
        predictions = self.model.model.predict(processed_test_images, verbose=1)
        predictions = predictions.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(test_steering, predictions)
        mae = mean_absolute_error(test_steering, predictions)
        r2 = r2_score(test_steering, predictions)
        
        # Additional metrics
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(test_steering - predictions))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'max_error': float(max_error),
            'num_samples': len(test_steering)
        }
        
        self.logger.info(f"Evaluation results: {metrics}")
        
        return metrics, predictions
        
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot
        """
        if self.training_history is None:
            self.logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.training_history.history['loss'], label='Training Loss')
        axes[0].plot(self.training_history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        if 'mae' in self.training_history.history:
            axes[1].plot(self.training_history.history['mae'], label='Training MAE')
            axes[1].plot(self.training_history.history['val_mae'], label='Validation MAE')
            axes[1].set_title('Model MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    def plot_predictions(self, test_data, predictions, save_path=None, num_samples=100):
        """
        Plot prediction results
        
        Args:
            test_data: Test data
            predictions: Model predictions
            save_path: Path to save plot
            num_samples: Number of samples to plot
        """
        test_images, test_steering = test_data
        
        # Select random samples
        indices = np.random.choice(len(test_steering), min(num_samples, len(test_steering)), replace=False)
        sample_steering = test_steering[indices]
        sample_predictions = predictions[indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scatter plot
        axes[0, 0].scatter(sample_steering, sample_predictions, alpha=0.6)
        axes[0, 0].plot([sample_steering.min(), sample_steering.max()], 
                       [sample_steering.min(), sample_steering.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Steering Angle')
        axes[0, 0].set_ylabel('Predicted Steering Angle')
        axes[0, 0].set_title('Actual vs Predicted Steering Angles')
        axes[0, 0].grid(True)
        
        # Error distribution
        errors = sample_predictions - sample_steering
        axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Error Distribution')
        axes[0, 1].grid(True)
        
        # Time series plot
        axes[1, 0].plot(sample_steering[:50], label='Actual', linewidth=2)
        axes[1, 0].plot(sample_predictions[:50], label='Predicted', linewidth=2)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Steering Angle')
        axes[1, 0].set_title('Steering Angle Time Series')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Absolute error
        abs_errors = np.abs(errors)
        axes[1, 1].plot(abs_errors)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Absolute Prediction Error')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Predictions plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    def run_full_training_pipeline(self, data_dir, model_name="pilotnet"):
        """
        Run complete training pipeline
        
        Args:
            data_dir: Directory containing training data
            model_name: Name for the model
            
        Returns:
            dict: Training results
        """
        self.logger.info("Starting full training pipeline...")
        
        # Load and preprocess data
        train_data, val_data, test_data = self.load_and_preprocess_data(data_dir)
        
        # Create model
        self.create_model()
        
        # Train model
        history = self.train_model(train_data, val_data, model_name)
        
        # Evaluate model
        metrics, predictions = self.evaluate_model(test_data)
        
        # Create visualizations
        history_plot_path = os.path.join(self.model_save_path, f"{model_name}_training_history.png")
        self.plot_training_history(history_plot_path)
        
        predictions_plot_path = os.path.join(self.model_save_path, f"{model_name}_predictions.png")
        self.plot_predictions(test_data, predictions, predictions_plot_path)
        
        # Save results
        results = {
            'model_name': model_name,
            'training_samples': len(train_data[0]),
            'validation_samples': len(val_data[0]),
            'test_samples': len(test_data[0]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'evaluation_metrics': metrics,
            'training_time': datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.model_save_path, f"{model_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Training pipeline completed successfully")
        self.logger.info(f"Results saved to {results_path}")
        
        return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PilotNet model')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing training data')
    parser.add_argument('--model_name', type=str, default='pilotnet',
                       help='Name for the model')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PilotNetTrainer()
    
    # Run training pipeline
    try:
        results = trainer.run_full_training_pipeline(
            args.data_dir, 
            args.model_name
        )
        print(f"Training completed successfully!")
        print(f"Final validation loss: {results['final_val_loss']:.4f}")
        print(f"Test R2 score: {results['evaluation_metrics']['r2_score']:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise