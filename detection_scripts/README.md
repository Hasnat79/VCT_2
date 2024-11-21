# AI-Generated Image Detection Scripts
This folder contains various inference scripts for detecting AI-generated images using different models and approaches. Each script implements a unique detection method and provides evaluation metrics.


### 1. DE-FAKE Detector (`de-fake_inference.py`)
- Multi-modal approach combining CLIP and BLIP models
- Uses neural network architecture for feature fusion
- Processes both images and generated captions
- Provides comprehensive metrics including accuracy, precision, recall, and F1 score
- Generates confusion matrix visualization

### 2. DIRE Detector (`dire_inference.py`)
- Based on ResNet architecture
- Provides probability scores for synthetic image detection
- Includes batch processing capabilities
- Supports CPU and GPU inference

### 3. DM Image Detection (`dm_image_detection_inference.py`)
- Specialized for diffusion model generated images
- Calculates multiple evaluation metrics
- Supports batch processing of images
- Outputs detailed CSV reports with metrics

### 4. DRCT Detection (`drct_inference.sh`)
- Shell script for running DRCT model inference
- Supports multiple model architectures including ConvNext and CLIP-ViT
- Configurable batch processing
- Handles both real and AI-generated image directories

### 5. FAKE Image Detection (`fake_image_detection_inference.py`)
- Implements multiple model architectures
- Supports various dataset types
- Provides comprehensive evaluation metrics
- Includes data augmentation capabilities

### 6. GAN Image Detection (`gan_image_detection_inference.py`)
- Specialized for detecting GAN-generated images
- Uses ensemble of EfficientNet models
- Implements patch-based analysis
- Provides probability scores and evaluation metrics

### 7. LASTED Detection (`lasted_inference.py`)
- Advanced clustering-based detection approach
- Implements cosine similarity metrics
- Supports both image and feature-based analysis
- Provides detailed evaluation metrics

### 8. NPR Detection (`npr_inference.py`)
- ResNet-based detection model
- Supports multiple dataset evaluation
- Provides comprehensive metrics output
- Includes size-based image analysis

### 9. OCC-CLIP Detection (`occ_clip_inference.py`)
- CLIP-based detection approach
- Batch processing capabilities
- Generates AUC and accuracy metrics
- Support for multiple image formats

### 10. RINE Detection (`rine_inference.py`)
- Real/fake image classification
- CSV output for detailed analysis
- Provides probability scores
- Calculates precision, recall, and F1 scores

### 11. SSH Detection (`ssh_inference.py`)
- SSP model implementation
- Supports both CPU and GPU inference
- Provides comprehensive evaluation metrics
- Includes data augmentation options

### 12. AIDE Detection (`aide_inference.py`)
- Multi-model ensemble approach
- Support for ResNet and ConvNext architectures
- Comprehensive evaluation pipeline
- Detailed logging and metrics tracking

### 13. CNN Detection (`cnn_detection_inference.py`)
- ResNet50-based detection
- Support for batch processing
- Size-based image analysis capabilities
- Comprehensive metrics output

### 14. Deepfake Detection (`deepfake_detection_inference.py`)
- XDNNClassifier implementation
- Support for feature extraction
- Visualization capabilities
- Detailed evaluation metrics

## Output Metrics
Most scripts provide the following evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Average Precision
## Dependencies
Common dependencies across scripts:
- PyTorch
- torchvision
- numpy
- scikit-learn
- PIL
- tqdm
- pandas
