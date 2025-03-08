# Vision Attribute Index (VAI) Script

This folder contains scripts for calculating the Vision Attribute Index (VAI), a method for analyzing and scoring image attributes. The implementation consists of two main components: feature extraction and score calculation.

## Components

### 1. Feature Extraction (`feature_extraction_agid.py`)

This script extracts seven key visual attributes from images:

- **Texture Complexity (TCI)**: Measures the intricacy of texture patterns using Local Binary Patterns (LBP)
- **Color Distribution Consistency (CDC)**: Evaluates the uniformity of color distribution using HSV color space
- **Object Coherence (OCI)**: Analyzes edge coherence using Canny edge detection
- **Contextual Relevance (CR)**: Measures spatial relationships using Sobel gradients
- **Image Smoothness (SMO)**: Calculates image smoothness using Laplacian variance
- **Image Sharpness (SHP)**: Determines image sharpness using Gaussian blur difference
- **Image Contrast (CON)**: Measures the standard deviation of pixel intensities

#### Functions

```python
calculate_texture_complexity(image, image_path)
calculate_color_distribution_consistency(image)
calculate_object_coherence(image)
calculate_contextual_relevance(image)
calculate_image_smoothness(image)
calculate_image_sharpness(image)
calculate_image_contrast(image)
```

The script processes all images in a specified folder and outputs a CSV file containing the calculated features for each image.

### 2. Score Calculation (`agid_score_calculation.py`)

This script processes the extracted features to calculate the final VAI score:

**i. Calculates the ADI (Attribute Deviation Index) score using:**

- Minimum values (L)
- Mean values (μ)
- Delta weights (δ)

**ii. Scales the ADI score to a 0-100 range using min-max normalization**

#### Score Formula

```python
ADI_Score = 100/N² * Σ(δᵢ * (xᵢ - Lᵢ)/(1 - μᵢ))
Scaled_Score = 100 * (ADI - ADI_min)/(ADI_max - ADI_min)
```

## Usage

### 1. First, run feature extraction:

```bash
python feature_extraction_agid.py
```

### 2. Then calculate the VAI scores:

```bash
python agid_score_calculation.py
```

## Output

- Feature extraction generates a CSV file with all extracted features
- Score calculation produces a final CSV with filename and scaled VAI scores
- The script also outputs the average VAI score for the dataset
