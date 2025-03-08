# 📁 Data
This directory contains the datasets used for AI-generated image detection. The data includes both real images from Twitter and COCO, along with their AI-generated counterparts.

# 🗂️ Directory Structure
The data is organized into the following structure:
```
data/
├── saved_images_twitter/              
│   ├── twitter_image/                 # Real Twitter images
│   ├── sd3_image/                     # Stable Diffusion 3 generated
│   ├── sd21_image/                    # Stable Diffusion 2.1 generated
│   ├── sdxl_image/                    # Stable Diffusion XL generated
│   ├── dalle_image/                   # DALL-E generated
│   └── captions.xlsx                  # Image captions
│
├── saved_images_coco/                 
│   ├── coco_image/                    # Real COCO images
│   ├── sd3_image/                     # Stable Diffusion 3 generated
│   ├── sd21_image/                    # Stable Diffusion 2.1 generated
│   ├── sdxl_image/                    # Stable Diffusion XL generated
│   ├── dalle_image/                   # DALL-E generated
│   ├── midjourney_image/             # Midjourney generated
│   └── captions.xlsx                  # Image captions
└── 
```

# 💾 Download Datasets
To download the datasets, use the provided scripts:\
```
# Download Twitter dataset
python hf_data_download_twitter.py

# Download COCO dataset
python hf_data_download_coco.py

# Verify the data structure
tree ./
```

