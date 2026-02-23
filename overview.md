# Dataset Split Overview

## Summary Table
```
┌──────────────┬───────────────┬──────────────┬──────────────┬───────────────┐
│    Aspect    │ full-training │   full-eye   │     main     │   segmented   │
├──────────────┼───────────────┼──────────────┼──────────────┼───────────────┤
│ Size         │ Largest (11K) │ Medium (1K)  │ Medium (1K)  │ Medium (1K)   │
├──────────────┼───────────────┼──────────────┼──────────────┼───────────────┤
│ Augmentation │ ✓ Yes         │ ✗ No         │ ✗ No         │ ✗ No          │
├──────────────┼───────────────┼──────────────┼──────────────┼───────────────┤
│ Datasets     │ Single        │ Mixed        │ India/Italy  │ India/Italy   │
├──────────────┼───────────────┼──────────────┼──────────────┼───────────────┤
│ Processing   │ Raw           │ Raw          │ Raw          │ ROI Segmented │
├──────────────┼───────────────┼──────────────┼──────────────┼───────────────┤
│ Use Case     │ Full training │ Multi-source │ Linear probe │ ROI-focused   │
└──────────────┴───────────────┴──────────────┴──────────────┴───────────────┘
```

## Full-Training (Largest Dataset)
- Source: `Dataset/Conjuctiva/Training/`
  - `Anemic/` (with augmentations: `aug3`, `aug15`, etc.)
  - `Non-Anemic/` (with augmentations)
- ~11,608 images
- Contains augmented images
- Used for full model training with data augmentation

## Full-Eye (Multi-Dataset Combination)
- Sources:
  - `Dataset/dataset anemia/` (Italy, India)
  - `Dataset/CP-AnemiC dataset/` (different source)
- ~1,022 images
- Combines multiple datasets
- Contains different eye regions: `forniceal`, `palpebral`, `forniceal_palpebral`
- Has extra column `,400` (likely bounding-box or metadata column)

## Main (Curated India/Italy Dataset)
- Source: `Dataset/dataset anemia/`
  - `Anemia/india_*`
  - `Anemia/italy_*`
  - `Non-Anemia/italy_*`
- ~1,003 images
- Clean subset (no augmentation)
- Original eye images (no segmentation)
- Used for optimized linear probe experiments

## Segmented (ROI-Extracted Dataset)
- Source: `Dataset/dataset anemia/` (segmented)
- ~1,003 images (same count as optimized)
- ROI-segmented images (eye region extracted)
- Same structure as optimized, but segmented/cropped for conjunctiva focus
