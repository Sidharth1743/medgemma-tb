# Path Foundation Model Card

**Author**: Google
**Version**: 1.0.0
**Created**: 2023-12-19

---

## Description

Path Foundation is a self-supervised foundation model for histopathology. It produces 384-dimensional embeddings from 224×224 pixel H&E image patches, enabling efficient downstream classifier training with reduced data and compute requirements.

**Research paper**: [Domain-specific optimization and diverse evaluation of self-supervised models for histopathology](https://arxiv.org/abs/2312.xxxxx)

---

## Resources

| Resource | Link |
|----------|------|
| Health AI Developer Foundations arXiv article | [Link](https://arxiv.org/) |
| Google Cloud Model Garden | Path Foundation |
| Hugging Face | [google/path-foundation](https://huggingface.co/google/path-foundation) |
| GitHub repository | [path-foundation](https://github.com/google-health/path-foundation) |
| Quick start notebook | `notebooks/quick_start` |
| Support | Contact documentation |
| Terms of use | Health AI Developer Foundations terms of use |

---

## Quick Start

```python
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download, from_pretrained_keras
import tensorflow as tf
import numpy as np

# Download a test image
hf_hub_download(
    repo_id="google/path-foundation",
    filename='Test.png',
    local_dir='.'
)

# Load and preprocess image
img = PILImage.open("Test.png").crop((0, 0, 224, 224)).convert('RGB')
tensor = tf.cast(tf.expand_dims(np.array(img), axis=0), tf.float32) / 255.0

# Load model from Hugging Face
loaded_model = from_pretrained_keras("google/path-foundation")

# Run inference
infer = loaded_model.signatures["serving_default"]
embeddings = infer(tf.constant(tensor))
embedding_vector = embeddings['output_0'].numpy().flatten()
```

---

## Examples

- **Quick start**: Run locally with Hugging Face weights in Colab
- **DICOM linear classifier**: Train with Google Cloud DICOM Store data
- **GCS linear classifier**: Train with Google Cloud Storage data

---

## Architecture

- **Base architecture**: ViT-S (Vision Transformer Small)
- **Training method**: Masked Siamese Networks across magnifications
- **Domain-specific tuning**: Optimized for histopathology

---

## Technical Specifications

| Property | Value |
|----------|-------|
| Model type | ViT-S |
| Version | 1.0.0 |
| Input size | 224×224 pixels |
| Output dimension | 384 |
| Supported magnifications | 5x, 10x, 20x |

---

## Performance

### Linear Probe Evaluation

Evaluated across 11 benchmark tasks with 17 unique tissue types, spanning various magnifications and task types.

**Key metric**: 93% mean AUC (95% CI: [92.9, 93.8])

Additional results for slide-level tasks (tissue type classification, molecular findings) and fine-tuning with data titration are available in the manuscript.

---

## Inputs and Outputs

**Input**: 224×224 pixel H&E patch from Whole Slide Images (WSIs)

Path Foundation integrates with [EZ-WSI](https://github.com/google-health/EZ-WSI) for WSI processing.

**Output**: 384-dimensional floating-point embedding vector

---

## Dataset

### Training Data

- **Source**: The Cancer Genome Atlas (TCGA), accessed via [GDC Portal](https://portal.gdc.cancer.gov)
- **Images**: Hematoxylin and eosin (H&E) stained WSIs
- **Patches**: 60 million patches across 3 magnifications:
  - ~2 µm/pixel (5x)
  - ~1 µm/pixel (10x)
  - ~0.5 µm/pixel (20x)
- **Studies**: 32 TCGA solid tumor studies, including tumor and diverse non-tumor patches

### Labeling

Training used self-supervised learning—no supervised labels required. Downstream task evaluation labels came from pathologist annotations or slide-level metadata.

### Data Citations

The Path Foundation results are based on data from the TCGA Research Network.

### References

- Benjordi, B. et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. *JAMA* (2017).
- Jaroensri, R. et al. Deep learning models for histologic grading of breast cancer and association with disease prognosis. *npj Breast Cancer* 8, 1–12 (2022).
- Liu, Y. et al. Artificial Intelligence-Based Breast Cancer Nodal Metastasis Detection: Insights Into the Black Box for Pathologists. *Arch. Pathol. Lab. Med.* 143 (2019).
- Lai, J. et al. Domain-specific optimization and diverse evaluation of self-supervised models for histopathology. *arXiv* (2023).
- Nagpal, K. et al. Development and Validation of a Deep Learning Algorithm for Gleason Grading of Prostate Cancer From Biopsy Specimens. *JAMA Oncol* 6, 1372–1380 (2020).
- Nagpal, K. et al. Development and validation of a deep learning algorithm for improving Gleason scoring of prostate cancer. *npj Digital Medicine* 2, 1–10 (2019).
- Sadhwani, A. et al. Comparative analysis of machine learning approaches to classify tumor mutation burden in lung adenocarcinoma using histopathology images. *Sci. Rep.* 11, 1–11 (2021).
- Wulczyn, E. et al. Interpretable survival prediction for colorectal cancer using deep learning. *NPJ Digital Medicine* 4 (2021).
- Weng, W.H. et al. Multimodal Multitask Representation Learning for Pathology Biobank Metadata Prediction. *arXiv* (2019).

---

## Implementation

### Software

Training was performed using JAX, enabling efficient use of modern hardware including TPUs.

---

## Use Cases

### Intended Use

Path Foundation reduces data, compute, and expertise needed to develop histopathology AI models.

**Applications**:

- Cancer detection, classification, and grading
- Metadata prediction (stain, tissue type, specimen type)
- Quality assessment (imaging artifacts)
- Similar-image search
- Biomarker discovery for prognostic and predictive tasks

### Benefits

- Efficient AI development with minimal data and compute
- More generalizable models than training on limited datasets
- Rich, compressed representations of histopathology patches

### Limitations

- Validated on limited downstream tasks
- Trained on H&E images from specific scanners and regions
- May not generalize to other image types, populations, or scanners
- Task-specific validation required for downstream applications
- Only evaluated at 5x, 10x, and 20x magnifications
- Generates embeddings only—no predictions or diagnoses provided

---

## License

Path Foundation use is governed by the [Health AI Developer Foundations terms of use](https://).
