"""
Visualization Utilities for Lane Keep Assist System
Functions for visualizing model activations, training progress, and system performance
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import STEERING, MODEL


def visualize_model_activations(model, image, layer_names=None, save_path=None):
    """
    Visualize CNN layer activations
    
    Args:
        model: Trained PilotNet model
        image: Input image
        layer_names: List of layer names to visualize
        save_path: Path to save visualization
    """
    if layer_names is None:
        layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5']
    
    # Get layer outputs
    layer_outputs = model.get_layer_outputs(image, layer_names)
    
    # Create subplots
    fig, axes = plt.subplots(len(layer_names), 8, figsize=(20, len(layer_names) * 3))
    if len(layer_names) == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer_name in enumerate(layer_names):
        activations = layer_outputs[layer_name][0]  # First image in batch
        
        # Show first 8 feature maps
        for j in range(min(8, activations.shape[-1])):
            ax = axes[i, j]
            activation = activations[:, :, j]
            
            # Normalize activation for visualization
            activation = (activation - activation.min()) / (activation.max() - activation.min())
            
            ax.imshow(activation, cmap='viridis')
            ax.set_title(f'{layer_name} - Filter {j+1}')
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_filter_weights(model, layer_name, save_path=None):
    """
    Visualize CNN filter weights
    
    Args:
        model: Trained PilotNet model
        layer_name: Name of the layer to visualize
        save_path: Path to save visualization
    """
    layer = model.model.get_layer(layer_name)
    weights = layer.get_weights()[0]  # Get kernel weights
    
    # Normalize weights
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Create subplots
    num_filters = weights.shape[-1]
    cols = 8
    rows = (num_filters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for i in range(num_filters):
        ax = axes[i]
        
        # For first layer, show RGB channels
        if weights.shape[-2] == 3:
            filter_img = weights[:, :, :, i]
        else:
            # For other layers, show first channel
            filter_img = weights[:, :, 0, i]
        
        ax.imshow(filter_img, cmap='viridis')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Filter Weights - {layer_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_steering_angle_distribution(steering_angles, bins=50, save_path=None):
    """
    Plot steering angle distribution with statistics
    
    Args:
        steering_angles: Array of steering angles
        bins: Number of histogram bins
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Main histogram
    axes[0, 0].hist(steering_angles, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Steering Angle (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Steering Angle Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add mean and std lines
    mean_angle = np.mean(steering_angles)
    std_angle = np.std(steering_angles)
    axes[0, 0].axvline(mean_angle, color='red', linestyle='--', label=f'Mean: {mean_angle:.2f}°')
    axes[0, 0].axvline(mean_angle + std_angle, color='orange', linestyle='--', alpha=0.7, label=f'±1 STD')
    axes[0, 0].axvline(mean_angle - std_angle, color='orange', linestyle='--', alpha=0.7)
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(steering_angles, vert=True)
    axes[0, 1].set_ylabel('Steering Angle (degrees)')
    axes[0, 1].set_title('Steering Angle Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_angles = np.sort(steering_angles)
    cumulative = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)
    axes[1, 0].plot(sorted_angles, cumulative, linewidth=2)
    axes[1, 0].set_xlabel('Steering Angle (degrees)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics table
    stats_data = {
        'Statistic': ['Count', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Q1', 'Q3'],
        'Value': [
            len(steering_angles),
            f'{np.mean(steering_angles):.3f}°',
            f'{np.std(steering_angles):.3f}°',
            f'{np.min(steering_angles):.3f}°',
            f'{np.max(steering_angles):.3f}°',
            f'{np.median(steering_angles):.3f}°',
            f'{np.percentile(steering_angles, 25):.3f}°',
            f'{np.percentile(steering_angles, 75):.3f}°'
        ]
    }
    
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=[[stat, val] for stat, val in zip(stats_data['Statistic'], stats_data['Value'])],
                           colLabels=['Statistic', 'Value'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_metrics(history, save_path=None):
    """
    Plot comprehensive training metrics
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training and validation MAE
    if 'mae' in history:
        axes[0, 1].plot(history['mae'], label='Training MAE', linewidth=2)
        axes[0, 1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history:
        axes[1, 0].plot(history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference
    loss_diff = np.array(history['val_loss']) - np.array(history['loss'])
    axes[1, 1].plot(loss_diff, linewidth=2, color='red')
    axes[1, 1].set_title('Validation - Training Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_analysis(actual, predicted, save_path=None):
    """
    Comprehensive prediction analysis plots
    
    Args:
        actual: Actual steering angles
        predicted: Predicted steering angles
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Scatter plot
    axes[0, 0].scatter(actual, predicted, alpha=0.6, s=20)
    axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Steering Angle (degrees)')
    axes[0, 0].set_ylabel('Predicted Steering Angle (degrees)')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predicted)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residual plot
    residuals = predicted - actual
    axes[0, 1].scatter(actual, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Actual Steering Angle (degrees)')
    axes[0, 1].set_ylabel('Residual (Predicted - Actual)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].axvline(0, color='red', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('Residual (degrees)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Time series comparison (first 200 points)
    n_points = min(200, len(actual))
    indices = np.arange(n_points)
    axes[1, 0].plot(indices, actual[:n_points], label='Actual', linewidth=2)
    axes[1, 0].plot(indices, predicted[:n_points], label='Predicted', linewidth=2)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Steering Angle (degrees)')
    axes[1, 0].set_title('Time Series Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Absolute error over time
    abs_errors = np.abs(residuals)
    axes[1, 1].plot(abs_errors[:n_points], linewidth=1, color='red')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Absolute Error (degrees)')
    axes[1, 1].set_title('Absolute Error Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error statistics
    mae = np.mean(abs_errors)
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    stats_text = f"""
    Error Statistics:
    MAE: {mae:.3f}°
    MSE: {mse:.3f}°²
    RMSE: {rmse:.3f}°
    R²: {r2:.3f}
    
    Max Error: {np.max(abs_errors):.3f}°
    Min Error: {np.min(abs_errors):.3f}°
    Std Error: {np.std(residuals):.3f}°
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_performance_dashboard(model, test_data, predictions, history, save_path=None):
    """
    Create a comprehensive performance dashboard
    
    Args:
        model: Trained model
        test_data: Test dataset
        predictions: Model predictions
        history: Training history
        save_path: Path to save dashboard
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['loss'], label='Training', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prediction scatter
    ax2 = fig.add_subplot(gs[0, 1])
    test_images, test_steering = test_data
    ax2.scatter(test_steering, predictions, alpha=0.6, s=20)
    ax2.plot([test_steering.min(), test_steering.max()], 
             [test_steering.min(), test_steering.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Actual vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = fig.add_subplot(gs[0, 2])
    errors = predictions - test_steering
    ax3.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', lw=2)
    ax3.set_xlabel('Error (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Steering angle distribution
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(test_steering, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Steering Angle (degrees)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Test Data Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Model architecture visualization
    ax5 = fig.add_subplot(gs[1, 0])
    layers = ['Input', 'Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Dense1', 'Dense2', 'Dense3', 'Output']
    y_pos = np.arange(len(layers))
    ax5.barh(y_pos, [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(layers)
    ax5.set_xlabel('Relative Size')
    ax5.set_title('Model Architecture')
    
    # Performance metrics
    ax6 = fig.add_subplot(gs[1, 1])
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(test_steering, predictions)
    mse = mean_squared_error(test_steering, predictions)
    r2 = r2_score(test_steering, predictions)
    
    metrics = ['MAE', 'MSE', 'R²']
    values = [mae, mse, r2]
    colors = ['green', 'orange', 'blue']
    
    bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
    ax6.set_ylabel('Value')
    ax6.set_title('Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Time series comparison
    ax7 = fig.add_subplot(gs[1, 2:])
    n_points = min(100, len(test_steering))
    indices = np.arange(n_points)
    ax7.plot(indices, test_steering[:n_points], label='Actual', linewidth=2)
    ax7.plot(indices, predictions[:n_points], label='Predicted', linewidth=2)
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('Steering Angle (degrees)')
    ax7.set_title('Time Series Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Residual analysis
    ax8 = fig.add_subplot(gs[2, :2])
    ax8.scatter(test_steering, errors, alpha=0.6, s=20)
    ax8.axhline(y=0, color='red', linestyle='--', lw=2)
    ax8.set_xlabel('Actual Steering Angle (degrees)')
    ax8.set_ylabel('Residual (Predicted - Actual)')
    ax8.set_title('Residual Analysis')
    ax8.grid(True, alpha=0.3)
    
    # Summary statistics
    ax9 = fig.add_subplot(gs[2, 2:])
    summary_text = f"""
    Model Performance Summary:
    
    Dataset Size: {len(test_steering)} samples
    
    Accuracy Metrics:
    • Mean Absolute Error: {mae:.3f}°
    • Mean Squared Error: {mse:.3f}°²
    • Root Mean Squared Error: {np.sqrt(mse):.3f}°
    • R² Score: {r2:.3f}
    
    Error Analysis:
    • Max Error: {np.max(np.abs(errors)):.3f}°
    • Min Error: {np.min(np.abs(errors)):.3f}°
    • Error Standard Deviation: {np.std(errors):.3f}°
    
    Training:
    • Final Training Loss: {history['loss'][-1]:.4f}
    • Final Validation Loss: {history['val_loss'][-1]:.4f}
    • Training Epochs: {len(history['loss'])}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax9.axis('off')
    
    plt.suptitle('Lane Keep Assist System - Performance Dashboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_preprocessing_pipeline(processor, image_path, save_path=None):
    """
    Visualize the complete preprocessing pipeline
    
    Args:
        processor: ImageProcessor instance
        image_path: Path to test image
        save_path: Path to save visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create preprocessing steps
    steps = []
    
    # Original image
    steps.append(('Original', cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    
    # Cropped image
    height, width = image.shape[:2]
    cropped = image[processor.crop_top:height-processor.crop_bottom, :]
    steps.append(('Cropped', cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))
    
    # Resized image
    resized = cv2.resize(cropped, (processor.target_width, processor.target_height))
    steps.append(('Resized', cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
    
    # YUV conversion
    if processor.yuv_conversion:
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        steps.append(('YUV', yuv))
    
    # Processed image
    processed = processor.preprocess_image(image, augment=False)
    if processor.normalize:
        processed = (processed * 255).astype(np.uint8)
    steps.append(('Processed', processed))
    
    # Augmented image
    augmented = processor.preprocess_image(image, augment=True)
    if processor.normalize:
        augmented = (augmented * 255).astype(np.uint8)
    steps.append(('Augmented', augmented))
    
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (title, img) in enumerate(steps):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(steps), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Generate sample data
    np.random.seed(42)
    steering_angles = np.random.normal(0, 5, 1000)
    steering_angles = np.clip(steering_angles, -25, 25)
    
    # Test steering angle distribution plot
    plot_steering_angle_distribution(steering_angles, save_path='test_distribution.png')
    print("Created test distribution plot")
    
    # Test prediction analysis
    predicted = steering_angles + np.random.normal(0, 1, len(steering_angles))
    plot_prediction_analysis(steering_angles, predicted, save_path='test_prediction_analysis.png')
    print("Created test prediction analysis plot")
    
    print("Visualization utilities test completed!")