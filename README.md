# Visual Counter Turing Test (VCT²)
## Overview
We introduce the Visual Counter Turing Test (VCT^2), a benchmark of approximately 130K images with prompts sourced from New York Times tweets and MS COCO captions. VCT2 highlights the limitations of current AGID techniques in detecting AI-generated images. To address the growing need for evaluation frameworks, we propose the Visual AI Index (VAI), which assesses generated images on texture complexity, color distribution, and object coherence, establishing a new standard for evaluating image-generation AI.

## Key Components:
## Dataset: ~130K images synthesized via state-of-the-art (SoTA) text-to-image generation models: <br />
Stable Diffusion 2.1 <br />
Stable Diffusion 3 <br />
Stable Diffusion XL <br />
DALL·E-3 <br />
Midjourney 6 <br />

#### *We make our datasets publicly available which can be accessed at:* 
COCO: https://huggingface.co/datasets/anonymous1233/COCO_AI <br />
Twitter: https://huggingface.co/datasets/anonymous1233/twitter_AI <br />

## Dual-Source Prompts:
VCT^2 includes two sets of prompts sourced from tweets by the New York Times Twitter account and captions from the MS COCO dataset.

## Visual AI Index

We introduce **Visual AI Index (VAI)**, a standardized metric to evaluate AI-generated images. VAI assesses generative models across key metrics:

- **Texture Complexity**: Measures texture variety using LBP entropy.
- **Color Distribution**: Analyzes variability in HSV histograms.
- **Object Coherence**: Evaluates edge clarity and boundary consistency.
- **Contextual Relevance**: Assesses edge strength distribution.
- **Image Quality**: Includes *smoothness*, *sharpness*, and *contrast* metrics.
