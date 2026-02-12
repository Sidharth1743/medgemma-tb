# MedSigLIP: Comprehensive Technical Report

## Executive Summary

MedSigLIP is a **400M-parameter medically-tuned vision encoder** derived from SigLIP-400M. It serves as the visual backbone for MedGemma and can also be used as a standalone encoder for medical image classification and retrieval tasks. MedSigLIP achieves performance comparable to or better than specialized medical image encoders across multiple domains.

---

## 1. Model Overview

| Attribute | Details |
|-----------|---------|
| **Model Name** | MedSigLIP |
| **Base Architecture** | SigLIP-400M (Zhai et al., 2023) |
| **Parameters** | 400 Million |
| **Input Resolution** | 448×448 pixels (released version) / 896×896 (MedGemma 4B) |
| **Model Type** | Vision-Language Encoder (Contrastive Learning) |
| **Primary Use** | Medical image encoding, zero-shot classification, linear probe classification |

### Key Characteristics
- **Multi-domain expertise**: Trained on radiology, dermatology, ophthalmology, and histopathology data
- **Dual resolution capability**: The 448×448 and 896×896 versions share the same model weights with down-sampled positional embeddings for the lower resolution
- **Data-efficient**: Strong zero-shot performance with further improvements via linear probing

---

## 2. Architecture Details

### Base Architecture (SigLIP-400M)
- **Vision Encoder**: Based on the SigLIP architecture
- **Image Resolution**: 
  - Released MedSigLIP: 448×448 pixels
  - MedGemma 4B integration: 896×896 pixels
- **Pixel Normalization**: Values normalized to [-1, 1]
- **Text Encoder**: Shared contrastive learning approach with SigLIP

### Resolution Adaptation
```
896×896 encoder (MedGemma 4B) 
    ↓ (down-sampled positional embeddings)
448×448 encoder (Released MedSigLIP)
```

> **Note**: The 448×448 encoder was chosen for the standalone release to enable more efficient experimentation while maintaining compatibility with the 896×896 version used in MedGemma 4B.

---

## 3. Training Methodology

### 3.1 Training Data

MedSigLIP was trained on **over 33 million medical image-text pairs**:

| Dataset Category | Number of Examples | Description |
|-----------------|-------------------|-------------|
| **Histopathology** | ~32.5 million | Histopathology image patches with caption pairs |
| **Radiology** | ~635k | Various medical imaging modalities |
| **Dermatology** | 51,049 | Skin lesion images with labels |
| **Ophthalmology** | 184,852 | Retinal fundus images |
| **Total Medical Data** | **~33 million** | Combined medical image-text pairs |

### Specific Training Datasets

| Modality | Dataset | Examples | Training Stage |
|----------|---------|----------|----------------|
| **Radiology** | MIMIC-CXR | 231,483 | Vision, PT, RL |
| | Digital Knee X-ray | 1,469 | Vision, PT |
| | CT-US | 59,979 | Vision, PT |
| | MRI-US | 47,622 | Vision, PT |
| **Histopathology** | Internal histopathology | 32,550,599 | Vision, PT, RL |
| **Dermatology** | PAD-UFES-20 | 2,047 | Vision, PT |
| | Internal dermatology | 51,049 | Vision, PT, RL |
| **Ophthalmology** | EyePACS | 199,258 | Vision, PT, RL |
| **General Medical** | PMC | 41,853 | Vision, PT |

### 3.2 Training Strategy

#### Vision Encoder Enhancement Process
1. **Base Model**: Started with SigLIP-400M from Gemma 3
2. **Data Mixing**: Medical data mixed with 2% weight into original SigLIP training data (WebLI)
3. **Purpose**: Retain SigLIP's existing general-domain performance while adding medical expertise
4. **Training Duration**: Fine-tuned on ~33M medical image-text pairs

#### Key Training Decisions
- **Resolution Finding**: Many medical vision tasks worked well at 448×448 resolution
- **Compatibility**: Maintained 896×896 for MedGemma 4B consistency
- **Data Retention**: Kept original WebLI data to preserve general visual capabilities

---

## 4. Evaluation Methodology

### 4.1 Evaluation Tasks

MedSigLIP was evaluated on two primary tasks:

| Task | Description | Purpose |
|------|-------------|---------|
| **Zero-shot Classification** | Direct classification using text prompts without task-specific training | Measure baseline performance |
| **Linear Probe Classification** | Logistic regression on frozen embeddings with limited training data | Measure data-efficient learning |

### 4.2 Evaluation Domains

| Domain | Dataset | Task Type | Classes |
|--------|---------|-----------|---------|
| **Radiology (CXR)** | CheXpert, CXR14 | 13 findings classification | Binary (present/absent) |
| **Dermatology** | US-Derm MCQA | 79 skin conditions | Multi-class |
| **Ophthalmology** | EyePACS | Diabetic retinopathy grading | 5-class (None to Proliferative) |
| **Histopathology** | Multiple internal datasets | Various tissue/pathology tasks | Varies by task |

### 4.3 Zero-shot Evaluation Approach

```
For each class:
  1. Generate text prompts describing the condition
  2. Average embeddings for multiple prompts per class
  3. Calculate cosine similarity between image and text embeddings
  4. Apply softmax to get classification scores
  5. Calculate AUC for each condition
```

### 4.4 Linear Probe Evaluation Approach

```
1. Extract image embeddings from MedSigLIP (frozen, no text encoder)
2. Train logistic regression with SAGA solver on train set
3. Tune hyperparameters on validation set
4. Evaluate on test set
```

---

## 5. Results

### Visual Summary

![MedSigLIP Key Highlights](medsiglip_images/fig4_key_highlights.png)

*Figure: MedSigLIP Key Performance Highlights - Architecture, advantages, and performance summary*

---

### 5.1 Chest X-Ray (CXR) Zero-shot Results

![CXR Zero-shot Performance](medsiglip_images/fig1_cxr_zero_shot_comparison.png)

*Figure: MedSigLIP Zero-shot CXR Classification Performance vs HAI-DEF CXR Foundation (ELIXR)*

**Table: Zero-shot AUCs for Chest X-ray Findings**

| Finding | MedSigLIP (448×448) | HAI-DEF/ELIXR (1280×1280) | Difference |
|---------|---------------------|---------------------------|------------|
| Enlarged Cardiomediastinum | 0.858 | 0.800 | **+5.8%** |
| Cardiomegaly | 0.904 | 0.891 | +1.3% |
| Lung Opacity | 0.931 | 0.888 | **+4.3%** |
| Lung Lesion | 0.822 | 0.747 | **+7.5%** |
| Consolidation | 0.880 | 0.875 | +0.5% |
| Edema | 0.891 | 0.880 | +1.1% |
| Pneumonia | 0.864 | 0.881 | -1.7% |
| Atelectasis | 0.836 | 0.754 | **+8.2%** |
| Pneumothorax | 0.862 | 0.800 | **+6.2%** |
| Pleural Effusion | 0.914 | 0.930 | -1.6% |
| Pleural Other | 0.650 | 0.729 | -7.9% |
| **Fracture** | **0.708** | **0.637** | **+7.1%** |
| Support Devices | 0.852 | 0.894 | -4.2% |
| **Average** | **0.844** | **0.824** | **+2.0%** |

**Key Findings:**
- MedSigLIP achieved **2.0% higher average AUC** than ELIXR despite:
  - Lower image resolution (448×448 vs 1280×1280)
  - Multi-domain expertise (not CXR-specialized)
- **Fracture detection** improved by **7.1%** - historically a difficult task
- Strong performance on atelectasis (+8.2%), lung lesion (+7.5%), and pneumothorax (+6.2%)

### 5.2 Chest X-Ray Linear Probe Results

**Figure: Data-Efficient Learning on 7 CXR Findings**

![CXR Data-Efficient Learning](medsiglip_images/fig3_data_efficient_learning.png)

*Figure: MedSigLIP Data-Efficient Learning on Chest X-Ray Findings - Linear Probe AUC vs Training Set Size*

**Key Observations:**
- MedSigLIP demonstrates strong performance with **≥512 training examples**
- Competitive with ELIXR (CXR-specialized model) across most findings
- Performance curves show MedSigLIP's data efficiency

### 5.3 Multi-Domain Performance Summary

![Multi-Domain Comparison](medsiglip_images/fig2_multidomain_comparison.png)

*Figure: MedSigLIP Multi-Domain Performance Comparison vs HAI-DEF Specialized Foundation Models*

### 5.4 Dermatology Results

**Table: Dermatology Classification Performance**

| Model | Resolution | Approach | AUC |
|-------|------------|----------|-----|
| **MedSigLIP** | 448×448 | Zero-shot | **0.851** |
| **MedSigLIP** | 448×448 | Linear Probe | **0.881** |
| HAI-DEF Derm Foundation | 224×224 | Linear Probe | 0.843 |

**Key Findings:**
- MedSigLIP zero-shot (0.851) already outperforms Derm Foundation linear probe (0.843)
- Linear probe improvement: +3.0% over zero-shot
- **Single model** (MedSigLIP) outperforms specialized dermatology model

### 5.4 Ophthalmology Results

**Table: Diabetic Retinopathy Classification**

| Approach | Resolution | AUC |
|----------|------------|-----|
| MedSigLIP Zero-shot | 448×448 | 0.759 |
| MedSigLIP Linear Probe | 448×448 | **0.857** |

**Key Findings:**
- Linear probe exceeds zero-shot by **9.8%**
- No HAI-DEF comparator available for ophthalmology
- Strong performance on 5-class DR grading task

### 5.5 Histopathology Results

**Table: Histopathology Classification Performance**

| Finding | MedSigLIP Zero-shot | MedSigLIP Linear Probe | HAI-DEF Path Foundation |
|---------|--------------------|------------------------|------------------------|
| Invasive Breast Cancer | 0.933 | 0.930 | 0.943 |
| Breast NP | 0.721 | 0.727 | 0.758 |
| Breast TF | 0.780 | 0.790 | 0.832 |
| Cervical Dysplasia | 0.889 | 0.864 | 0.898 |
| Prostate Cancer (Needle) | 0.892 | 0.886 | 0.915 |
| Prostate Cancer (Radical) | 0.896 | 0.887 | 0.921 |
| TCGA Study Types | 0.922 | 0.970 | 0.964 |
| Tissue Types | 0.930 | 0.972 | 0.947 |
| **Average** | **0.870** | **0.878** | **0.897** |

**Key Findings:**
- MedSigLIP performs close to Path Foundation (1.9% gap on average)
- Path Foundation is specialized for histopathology only
- MedSigLIP is a **single multi-domain model**
- Strong linear probe performance on TCGA Study Types (0.970) and Tissue Types (0.972)

---

## 6. Comparison with HAI-DEF Foundation Models

### Summary Table: MedSigLIP vs Specialized Models

| Domain | MedSigLIP | HAI-DEF Specialized Model | Notes |
|--------|-----------|--------------------------|-------|
| **CXR** | 0.844 AUC (zero-shot) | 0.824 AUC (ELIXR) | MedSigLIP wins despite lower resolution |
| **Dermatology** | 0.881 AUC (linear probe) | 0.843 AUC (Derm Foundation) | MedSigLIP wins |
| **Ophthalmology** | 0.857 AUC (linear probe) | N/A | No comparator available |
| **Histopathology** | 0.878 AUC (linear probe) | 0.897 AUC (Path Foundation) | 1.9% gap, but single model |

**Key Advantage**: MedSigLIP is a **single model** covering all domains, while HAI-DEF requires separate specialized models for each domain.

---

## 7. Zero-shot Prompts Used for Evaluation

### 7.1 Chest X-Ray Prompts (Examples)

| Finding | Condition Absent Prompts | Condition Present Prompts |
|---------|------------------------|--------------------------|
| Atelectasis | "no atelectasis", "no acute cardiopulmonary process" | "adjacent atelectasis", "bibasilar atelectasis", "there is atelectasis" |
| Cardiomegaly | "heart size is normal", "cardiac size is within normal limits" | "mild cardiomegaly", "moderate cardiomegaly", "severe cardiomegaly" |
| Consolidation | "no focal consolidation", "normal study" | "alveolar consolidation", "densely consolidated", "lobe consolidation" |
| Fracture | "no acute cardiopulmonary process", "normal study" | "rib fractures", "rib fracture" |

### 7.2 Ophthalmology Prompts

| Severity | Prompt |
|----------|--------|
| No DR | "diabetic retinopathy severity: none" |
| Mild DR | "diabetic retinopathy severity: mild" |
| Moderate DR | "diabetic retinopathy severity: moderate" |
| Severe DR | "diabetic retinopathy severity: severe" |
| Proliferative DR | "diabetic retinopathy severity: proliferative" |

### 7.3 Histopathology Prompts (Example: Invasive Breast Cancer)

| Class | Example Prompts |
|-------|-----------------|
| Benign | "region of an HE histopathology image showing benign breast tissue", "HE-stained image demonstrating normal breast tissue architecture" |
| Invasive Carcinoma | "region of an HE histopathology image showing invasive breast carcinoma", "HE-stained region demonstrating features of invasive breast carcinoma" |
| DCIS | "region of an HE histopathology image showing ductal carcinoma in situ (DCIS)", "HE-stained image demonstrating abnormal cells confined within the breast duct" |

---

## 8. Key Advantages of MedSigLIP

### 8.1 Multi-Domain Capability
- **Single model** for radiology, dermatology, ophthalmology, and histopathology
- Eliminates need for separate specialized encoders
- Reduces deployment complexity

### 8.2 Data Efficiency
- Strong zero-shot performance without task-specific training
- Linear probing achieves competitive results with limited labeled data (≥512 examples)
- Suitable for scenarios with scarce annotations

### 8.3 Resolution Flexibility
- 448×448 resolution for efficient experimentation
- 896×896 resolution for MedGemma integration
- Same weights, just down-sampled positional embeddings

### 8.4 Performance vs Efficiency Trade-off
- **2.0% improvement** over CXR Foundation despite:
  - 8× lower resolution (448×448 vs 1280×1280)
  - Multi-domain training (not CXR-specialized)
- Outperforms specialized dermatology model

---

## 9. Use Cases

### 9.1 Standalone Applications
1. **Zero-shot Image Classification**: Direct classification without training data
2. **Linear Probe Classification**: Data-efficient adaptation with limited labels
3. **Image Retrieval**: Find similar medical images using embedding similarity
4. **Multi-modal Search**: Combine image and text embeddings for retrieval

### 9.2 Integration with MedGemma
- Powers the visual understanding capabilities of MedGemma 4B
- Enables medical VQA, report generation, and image-text reasoning
- Shared encoder ensures consistency across applications

---

## 10. Model Availability

| Resource | Link |
|----------|------|
| **Model Weights** | https://goo.gle/hai-def |
| **MedGemma Collection** | https://goo.gle/medgemma |
| **Documentation & Tutorials** | Available at MedGemma website |

---

## 11. Summary Statistics

### Performance Summary

| Metric | Value |
|--------|-------|
| **Parameters** | 400M |
| **Training Data** | ~33M medical image-text pairs |
| **CXR Zero-shot AUC** | 0.844 (avg across 13 findings) |
| **Dermatology Linear Probe AUC** | 0.881 (79 conditions) |
| **Ophthalmology Linear Probe AUC** | 0.857 (5-class DR) |
| **Histopathology Linear Probe AUC** | 0.878 (avg across 8 tasks) |

### Comparison Summary

| Comparison | MedSigLIP Advantage |
|------------|---------------------|
| vs CXR Foundation (ELIXR) | +2.0% AUC, 8× lower resolution |
| vs Derm Foundation | +3.8% AUC, single model |
| vs Path Foundation | -1.9% AUC, but multi-domain |

---

## 12. Conclusion

MedSigLIP represents a significant advancement in medical vision encoding by demonstrating that a **single, multi-domain model** can achieve performance comparable to or better than **specialized, domain-specific encoders**. Key contributions include:

1. **Unified Architecture**: One model for radiology, dermatology, ophthalmology, and histopathology
2. **Strong Baseline Performance**: Excellent zero-shot capabilities across all domains
3. **Data Efficiency**: Linear probing achieves competitive results with limited training data
4. **Practical Efficiency**: 448×448 resolution enables efficient experimentation without sacrificing performance
5. **Integration Ready**: Powers MedGemma's multimodal capabilities while available as a standalone encoder

MedSigLIP provides a strong foundation for medical image understanding applications, enabling both direct use for classification/retrieval and integration into larger vision-language systems.

---

## References

- Zhai et al. (2023). Sigmoid loss for language image pre-training. ICCV.
- Xu et al. (2023). ELIXR: Towards a general purpose x-ray artificial intelligence system.
- Kiraly et al. (2024). Health AI Developer Foundations.
- Gemma Team et al. (2025). Gemma 3 Technical Report.

---

*Report compiled from the MedGemma Technical Report (Google Research and Google DeepMind, 2025)*
