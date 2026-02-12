# MedSigLIP

MedSigLIP is a medical vision-language model that encodes medical images and text into a shared embedding space. It enables developers to accelerate healthcare AI application development.

## Overview

MedSigLIP contains a 400M-parameter vision encoder and a 400M-parameter text encoder. It supports 448×448 image resolution with up to 64 text tokens. The model was trained on de-identified medical image-text pairs from multiple modalities:

- Chest X-rays
- Dermatology images
- Ophthalmology images
- Histopathology slides
- CT and MRI volume slices

Each image was paired with associated descriptions or reports.

**Recommended use case**: Medical image interpretation applications that don't require text generation, such as classification and retrieval. For text generation tasks, use MedGemma instead.

## Common Use Cases

### Data-Efficient Classification

MedSigLIP serves as an excellent foundation for building medical image classifiers with minimal labeled data. Training a classifier on MedSigLIP embeddings requires significantly less data than training from scratch.

Once an image is encoded, the embedding can be reused across multiple classifiers with minimal additional compute.

**Note**: While MedSigLIP has histopathology pretraining, Path Foundation is recommended for digital pathology classification due to its lower compute requirements and comparable performance.

### Zero-Shot Classification

Leverage MedSigLIP's text encoder to obtain classification scores without any training data. Zero-shot classification compares image embeddings to positive and negative text prompts:

- Positive example: "pleural effusion present"
- Negative example: "normal X-ray"

This approach excels with limited data. As training data increases, data-efficient classification typically outperforms zero-shot.

### Semantic Image Retrieval

Rank medical images by relevance to a search query using the text encoder. Retrieve similar images by comparing their embeddings to the text embedding of your query.

### Fine-Tuning

Improve performance on existing tasks or adapt MedSigLIP to new tasks through fine-tuning.

## Resources

- [Data-efficient classifier notebook](https://colab.research.google.com/github/google-health/medsiglip/blob/main/notebooks/train_data_efficient_classifier.ipynb)
- [Getting started Colab notebook](https://colab.research.google.com/github/google-health/medsiglip)
