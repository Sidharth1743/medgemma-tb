# Path Foundation

Path Foundation is a machine learning model that generates embeddings from digital pathology images. These embeddings enable efficient development of AI models for histopathology tasks, requiring significantly less data and compute than training from scratch.

## Overview

Trained on large-scale pathology datasets, Path Foundation helps healthcare and life sciences organizations build AI models for pathology image analysis with minimal data.

**For detailed usage and training information, see the [Path Foundation model card](path-foundation-get.md).**

## Common Use Cases

### Data-Efficient Classification

Use Path Foundation embeddings to train classifiers with minimal labeled data. Once a tissue patch is encoded, the embedding can be reused across multiple classifiers with minimal additional compute.

**Applications include**:

- Tumor tissue identification and tissue class discrimination
- Tumor grading
- Biomarker detection for treatment response prediction
- Novel biomarker discovery through feature exploration
- Feature detection and classification on Whole Slide Images (WSIs)
- Clinical feature detection
- Tissue and stain type determination
- Pathology image quality assessment

**Example notebooks**:

- [Path Foundation linear classifier - Cloud Storage](https://colab.research.google.com/github/google-health/path-foundation)
- [Path Foundation linear classifier - Google Cloud DICOM](https://colab.research.google.com/github/google-health/path-foundation)

### Similar-Image Search

Find similar images within or between WSIs using embedding distances. Select reference patches, then identify and retrieve the most similar patches from any set of WSIs.

## Technical Summary

- **Input**: 224×224 pixel H&E patches from Whole Slide Images
- **Output**: 384-dimensional embedding vectors
- **Architecture**: ViT-S trained with Masked Siamese Networks
- **Performance**: 93% mean AUC across 11 histopathology benchmark tasks (95% CI: [92.9, 93.8])
- **Training data**: 60 million patches from TCGA across 3 magnifications (5x, 10x, 20x)

## Integration

Path Foundation integrates with [EZ-WSI](https://github.com/google-health/EZ-WSI), a digital pathology library for processing WSIs into patches.
