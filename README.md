# Visual Counter Turing Test (VCT²)

## Overview
The **Visual Counter Turing Test (VCT²)** is a benchmark dataset of approximately 130,000 images with prompts sourced from New York Times tweets and MS COCO captions. VCT² highlights the limitations of current AI-generated image detection (AGID) techniques in identifying AI-generated content. 

To address the growing need for evaluation frameworks, we introduce the **Visual AI Index (VAI)**, a novel metric designed to evaluate generative models on texture complexity, color distribution, and object coherence. This establishes a new standard for assessing image-generation AI.

---

## Key Components

### Dataset
VCT² comprises ~130,000 images synthesized using state-of-the-art (SoTA) text-to-image generation models:
- **Stable Diffusion 2.1**
- **Stable Diffusion 3**
- **Stable Diffusion XL**
- **DALL·E-3**
- **Midjourney 6**

The datasets are publicly available:
- **COCO Prompts Dataset:** [Hugging Face Link](https://huggingface.co/datasets/anonymous1233/COCO_AI)
- **Twitter Prompts Dataset:** [Hugging Face Link](https://huggingface.co/datasets/anonymous1233/twitter_AI)

### Dual-Source Prompts
VCT² includes two distinct sets of prompts:
1. **Twitter Prompts:** Extracted from tweets by the New York Times Twitter account.
2. **COCO Prompts:** Derived from captions in the MS COCO dataset.

---

## Visual AI Index (VAI)
The **Visual AI Index (VAI)** provides a standardized metric to evaluate AI-generated images based on multiple dimensions of quality and coherence:

- **Texture Complexity:** Measures texture variety using Local Binary Patterns (LBP) entropy.
- **Color Distribution:** Analyzes variability in HSV (Hue, Saturation, Value) histograms.
- **Object Coherence:** Evaluates edge clarity and boundary consistency.
- **Contextual Relevance:** Assesses the distribution of edge strength across the image.
- **Image Quality:** Includes metrics for *smoothness*, *sharpness*, and *contrast*.

The VAI serves as a comprehensive evaluation tool for benchmarking the capabilities of image-generation models.
