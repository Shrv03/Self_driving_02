"""
Image Preprocessing Module for Lane Keep Assist System
Handles image preprocessing, augmentation, and dataset preparation
"""

import cv2
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
from imgaug import augmenters as iaa
import imgaug as ia

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import IMAGE_PREPROCESSING, DATA_AUGMENTATION, STEERING, DATASET


class ImageProcessor:
    """
    Image processor for preprocessing driving images
    """
    
    def __init__(self):
        """Initialize image processor"""
        self.target_height = IMAGE_PREPROCESSING['target_height']
        self.target_width = IMAGE_PREPROCESSING['target_width']
        self.crop_top = IMAGE_PREPROCESSING['crop_top']
        self.crop_bottom = IMAGE_PREPROCESSING['crop_bottom']
        self.normalize = IMAGE_PREPROCESSING['normalize']
        self.yuv_conversion = IMAGE_PREPROCESSING['yuv_conversion']
        self.gaussian_blur = IMAGE_PREPROCESSING['gaussian_blur']
        self.blur_kernel_size = IMAGE_PREPROCESSING['blur_kernel_size']
        
        # Setup augmentation pipeline
        self.setup_augmentation()
        
    def setup_augmentation(self):
        """Setup image augmentation pipeline"""
        self.augmentation_pipeline = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),  # Brightness
            iaa.Sometimes(0.3, iaa.Affine(
                rotate=(-DATA_AUGMENTATION['rotation_range'], DATA_AUGMENTATION['rotation_range']),
                translate_percent={
                    "x": (-DATA_AUGMENTATION['width_shift_range'], DATA_AUGMENTATION['width_shift_range']),
                    "y": (-DATA_AUGMENTATION['height_shift_range'], DATA_AUGMENTATION['height_shift_range'])
                },
                scale=(1.0 - DATA_AUGMENTATION['zoom_range'], 1.0 + DATA_AUGMENTATION['zoom_range'])
            )),
            iaa.Sometimes(0.5, self.add_random_shadow),
        ])
        
    def preprocess_image(self, image, augment=False):
        """
        Preprocess image for PilotNet model
        
        Args:
            image: Input image (BGR format)
            augment: Whether to apply augmentation
            
        Returns:
            np.array: Preprocessed image
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop image to remove sky and car hood
        height, width = image.shape[:2]
        cropped = image[self.crop_top:height-self.crop_bottom, :]
        
        # Resize to target dimensions
        resized = cv2.resize(cropped, (self.target_width, self.target_height))
        
        # Apply Gaussian blur if enabled
        if self.gaussian_blur:
            resized = cv2.GaussianBlur(resized, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Convert to YUV color space (as used in PilotNet)
        if self.yuv_conversion:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        
        # Apply augmentation if requested
        if augment:
            resized = self.augmentation_pipeline(image=resized)
        
        # Normalize pixel values
        if self.normalize:
            resized = resized.astype(np.float32) / 255.0
        
        return resized
        
    def add_random_shadow(self, image):
        """
        Add random shadow to image
        
        Args:
            image: Input image
            
        Returns:
            np.array: Image with shadow
        """
        # Convert to HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        
        # Generate random shadow polygon
        height, width = image.shape[:2]
        
        # Random shadow vertices
        top_x = random.randint(0, width)
        top_y = random.randint(0, height // 2)
        bottom_x = random.randint(0, width)
        bottom_y = random.randint(height // 2, height)
        
        # Create shadow mask
        shadow_mask = np.zeros((height, width), dtype=np.uint8)
        vertices = np.array([[top_x, top_y], [bottom_x, bottom_y], 
                           [bottom_x + 50, bottom_y], [top_x + 50, top_y]], dtype=np.int32)
        cv2.fillPoly(shadow_mask, [vertices], 255)
        
        # Apply shadow effect
        shadow_ratio = random.uniform(0.3, 0.7)
        hls[:, :, 1][shadow_mask == 255] = hls[:, :, 1][shadow_mask == 255] * shadow_ratio
        
        # Convert back to RGB
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        
    def process_dataset(self, data_dir, output_dir=None, augment_data=True):
        """
        Process entire dataset
        
        Args:
            data_dir: Directory containing raw data
            output_dir: Directory to save processed data
            augment_data: Whether to apply data augmentation
            
        Returns:
            tuple: (images, steering_angles)
        """
        if output_dir is None:
            output_dir = os.path.join(data_dir, 'processed')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load steering data
        steering_file = os.path.join(data_dir, 'steering_log.csv')
        if not os.path.exists(steering_file):
            raise FileNotFoundError(f"Steering data not found: {steering_file}")
        
        df = pd.read_csv(steering_file)
        
        images = []
        steering_angles = []
        
        print(f"Processing {len(df)} samples...")
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} samples")
            
            # Load image
            image_path = row['image_path']
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            steering_angle = float(row['steering_angle'])
            
            # Process original image
            processed_image = self.preprocess_image(image, augment=False)
            images.append(processed_image)
            steering_angles.append(steering_angle)
            
            # Add augmented versions for data with significant steering
            if augment_data and abs(steering_angle) > STEERING['measurement_threshold']:
                # Add multiple augmented versions
                for _ in range(3):
                    augmented_image = self.preprocess_image(image, augment=True)
                    images.append(augmented_image)
                    steering_angles.append(steering_angle)
        
        print(f"Dataset processing complete. Total samples: {len(images)}")
        
        return np.array(images), np.array(steering_angles)
        
    def balance_dataset(self, images, steering_angles):
        """
        Balance dataset by steering angle distribution
        
        Args:
            images: Array of images
            steering_angles: Array of steering angles
            
        Returns:
            tuple: (balanced_images, balanced_steering_angles)
        """
        # Create bins for steering angles
        bins = np.linspace(STEERING['min_steering_angle'], 
                          STEERING['max_steering_angle'], 
                          DATASET['steering_bins'])
        
        # Digitize steering angles
        steering_bins = np.digitize(steering_angles, bins)
        
        # Find maximum samples per bin
        bin_counts = np.bincount(steering_bins)
        max_samples = max(DATASET['min_samples_per_angle'], 
                         int(np.median(bin_counts[bin_counts > 0])))
        
        balanced_images = []
        balanced_steering = []
        
        for bin_idx in range(1, len(bins)):
            # Get samples in this bin
            mask = steering_bins == bin_idx
            bin_images = images[mask]
            bin_steering = steering_angles[mask]
            
            if len(bin_images) == 0:
                continue
            
            # Sample with replacement if needed
            if len(bin_images) < max_samples:
                indices = np.random.choice(len(bin_images), max_samples, replace=True)
            else:
                indices = np.random.choice(len(bin_images), max_samples, replace=False)
            
            balanced_images.extend(bin_images[indices])
            balanced_steering.extend(bin_steering[indices])
        
        return np.array(balanced_images), np.array(balanced_steering)
        
    def split_dataset(self, images, steering_angles, train_ratio=None, val_ratio=None, test_ratio=None):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            images: Array of images
            steering_angles: Array of steering angles
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        if train_ratio is None:
            train_ratio = DATASET['train_ratio']
        if val_ratio is None:
            val_ratio = DATASET['val_ratio']
        if test_ratio is None:
            test_ratio = DATASET['test_ratio']
        
        # Shuffle data
        images, steering_angles = shuffle(images, steering_angles, random_state=42)
        
        total_samples = len(images)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        # Split data
        train_images = images[:train_end]
        train_steering = steering_angles[:train_end]
        
        val_images = images[train_end:val_end]
        val_steering = steering_angles[train_end:val_end]
        
        test_images = images[val_end:]
        test_steering = steering_angles[val_end:]
        
        return (train_images, train_steering), (val_images, val_steering), (test_images, test_steering)
        
    def visualize_preprocessing(self, original_image, save_path=None):
        """
        Visualize preprocessing steps
        
        Args:
            original_image: Original input image
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Cropped image
        height, width = original_image.shape[:2]
        cropped = original_image[self.crop_top:height-self.crop_bottom, :]
        axes[0, 1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Cropped Image')
        axes[0, 1].axis('off')
        
        # Resized image
        resized = cv2.resize(cropped, (self.target_width, self.target_height))
        axes[0, 2].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Resized Image')
        axes[0, 2].axis('off')
        
        # YUV conversion
        if self.yuv_conversion:
            yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
            axes[1, 0].imshow(yuv)
            axes[1, 0].set_title('YUV Image')
        else:
            axes[1, 0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('RGB Image')
        axes[1, 0].axis('off')
        
        # Preprocessed image
        processed = self.preprocess_image(original_image, augment=False)
        if self.normalize:
            processed = (processed * 255).astype(np.uint8)
        axes[1, 1].imshow(processed)
        axes[1, 1].set_title('Preprocessed Image')
        axes[1, 1].axis('off')
        
        # Augmented image
        augmented = self.preprocess_image(original_image, augment=True)
        if self.normalize:
            augmented = (augmented * 255).astype(np.uint8)
        axes[1, 2].imshow(augmented)
        axes[1, 2].set_title('Augmented Image')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
    def analyze_steering_distribution(self, steering_angles, save_path=None):
        """
        Analyze and visualize steering angle distribution
        
        Args:
            steering_angles: Array of steering angles
            save_path: Path to save visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Histogram of steering angles
        plt.subplot(2, 2, 1)
        plt.hist(steering_angles, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Steering Angle (degrees)')
        plt.ylabel('Frequency')
        plt.title('Steering Angle Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(steering_angles, vert=True)
        plt.ylabel('Steering Angle (degrees)')
        plt.title('Steering Angle Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_angles = np.sort(steering_angles)
        cumulative = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)
        plt.plot(sorted_angles, cumulative)
        plt.xlabel('Steering Angle (degrees)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.grid(True, alpha=0.3)
        
        # Statistics
        plt.subplot(2, 2, 4)
        stats_text = f"""
        Statistics:
        Mean: {np.mean(steering_angles):.3f}°
        Median: {np.median(steering_angles):.3f}°
        Std: {np.std(steering_angles):.3f}°
        Min: {np.min(steering_angles):.3f}°
        Max: {np.max(steering_angles):.3f}°
        
        Samples: {len(steering_angles)}
        Zero steering: {np.sum(np.abs(steering_angles) < 0.1)}/{len(steering_angles)}
        """
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='center', fontsize=10, fontfamily='monospace')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def batch_generator(images, steering_angles, batch_size=32, augment=True):
    """
    Generate batches of preprocessed images and steering angles
    
    Args:
        images: Array of images
        steering_angles: Array of steering angles
        batch_size: Batch size
        augment: Whether to apply augmentation
        
    Yields:
        tuple: (batch_images, batch_steering)
    """
    processor = ImageProcessor()
    
    while True:
        # Shuffle data
        images, steering_angles = shuffle(images, steering_angles)
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_steering = steering_angles[i:i+batch_size]
            
            # Preprocess batch
            processed_batch = []
            for img in batch_images:
                processed_img = processor.preprocess_image(img, augment=augment)
                processed_batch.append(processed_img)
            
            yield np.array(processed_batch), np.array(batch_steering)


if __name__ == "__main__":
    # Test image processing
    processor = ImageProcessor()
    
    # Test with a sample image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    processed = processor.preprocess_image(test_image, augment=False)
    print(f"Original shape: {test_image.shape}")
    print(f"Processed shape: {processed.shape}")
    
    # Test augmentation
    augmented = processor.preprocess_image(test_image, augment=True)
    print(f"Augmented shape: {augmented.shape}")
    
    # Test batch generator
    images = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
    steering = np.random.uniform(-25, 25, 100)
    
    gen = batch_generator(images, steering, batch_size=16)
    batch_img, batch_steering = next(gen)
    print(f"Batch images shape: {batch_img.shape}")
    print(f"Batch steering shape: {batch_steering.shape}")