# MRI Brain-print as Secure Hidden Biometric System

*A novel approach to biometric identification using cortical fold patterns from MRI scans*

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technical Specifications](#technical-specifications)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [File Structure](#file-structure)
- [References](#references)
- [License](#license)

## Project Overview

This project implements a cutting-edge biometric identification system using MRI-derived brain-prints, achieving **96.30% recognition accuracy**. Unlike traditional biometrics (fingerprints, iris scans), this "hidden biometric" leverages unique cortical fold patterns that are:
- Impossible to spoof or replicate
- Stable throughout adulthood
- Highly distinctive (even in identical twins)

**Core Innovation**: Extracts discriminative features from 3D brain MRI volumes using advanced image processing and deep learning techniques.

## Key Features

- 🧠 **Non-invasive biometric identification** using structural MRI scans
- 🔍 **Multi-stage processing pipeline**:
  - Advanced image denoising (Gaussian + Contraharmonic filtering)
  - Gabor wavelet feature extraction
  - PCA-based dimensionality reduction
- 🧠 **Deep learning architecture**:
  - Custom 6-layer CNN model
  - Batch normalization and dropout for regularization
  - Adamax optimization
- 📊 **Comprehensive evaluation**:
  - 96.30% identification accuracy
  - Minimal overfitting (train/val loss convergence)
  - Robustness analysis

## Technical Specifications

### Data Requirements
- **Input**: T1-weighted MRI volumes (OASIS dataset format)
- **Sample Size**: 220 healthy adults (18-55 years)
- **Slices Processed**: 20 slices per volume (120-139 range)

### Processing Pipeline
1. **Preprocessing**:
   - Gaussian denoising (3×3 kernel)
   - Histogram equalization
   - Contraharmonic mean filtering (Q=0.5)

2. **Feature Extraction**:
   - 16-orientation Gabor filter bank
   - Wavelet transform for texture analysis
   - PCA reduction to 64 components

3. **Classification**:
   - Custom CNN architecture (124.16MB parameters)
   - 10-epoch training (batch size=10)
   - Categorical cross-entropy loss

### Performance Metrics
| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 89.77% | 98.78% |
| Loss | 0.3055 | 0.0532 |
| Epochs | 10 | - |

## Installation

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA-enabled for TensorFlow)

### Dependencies
```bash
pip install -r requirements.txt
```
*requirements.txt*:
```
opencv-python==4.5.5
matplotlib==3.5.1
scikit-learn==1.0.2
tensorflow==2.8.0
numpy==1.22.3
```

## Usage

### 1. Data Preparation
Organize MRI slices in `Data/Mild Dementia/` following naming convention:
```
SubjectID_XXXX_sliceNNN.jpg
```

### 2. Running the Pipeline
Execute the Jupyter notebook:
```bash
jupyter notebook mri_biometric.ipynb
```

### 3. Custom Training
Modify these key parameters in the notebook:
```python
# In data processing:
n_components = 64          # PCA components
slice_range = (120, 140)   # MRI slice selection

# In model configuration:
epochs = 10                
batch_size = 10            
learning_rate = 0.001      
```

## Methodology

### Technical Approach
1. **Cortical Fold Extraction**:
   - Transform 3D pial surfaces → 2D curvature maps
   - Process 4 surfaces per subject (Fig 1.1)

2. **Image Enhancement**:
   ```math
   \text{Contraharmonic filter: } I_{out}(x,y)=\frac{\sum I(s,t)^{Q+1}}{\sum I(s,t)^Q}
   ```

3. **Gabor Wavelet Transform**:
   ```math
   g(x,y)=exp(-\frac{x^2}{2σ_x^2}-\frac{y^2}{2σ_y^2})·cos(2πf(x\cosθ+y\sinθ))
   ```
4. **CNN Architecture**:
```python
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (Type)                         ┃ Output Shape            ┃ Param #         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 248, 64, 32)     │            160  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 248, 64, 32)     │          4,128  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ batch_normalization (BatchNorm)      │ (None, 248, 64, 32)     │            128  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 124, 32, 32)     │              0  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 124, 32, 32)     │              0  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 124, 32, 64)     │          8,256  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 124, 32, 64)     │         16,448  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ batch_normalization_1 (BatchNorm)    │ (None, 124, 32, 64)     │            256  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 62, 16, 64)      │              0  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 62, 16, 64)      │              0  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 63488)           │              0  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 512)             │     32,506,368  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 512)             │              0  │
├──────────────────────────────────────┼─────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 21)              │         10,773  │
└──────────────────────────────────────┴─────────────────────────┴─────────────────┘

## Results

### Performance Analysis
![Accuracy Plot](https://i.imgur.com/accuracy_plot.png)
*Training vs validation accuracy over 10 epochs*

### Key Findings
- **96.30% recognition rate** in identification tasks
- Val accuracy peaks at **98.78%** (epoch 10)
- Rapid convergence (val_loss: 2.98 → 0.05)

### Comparative Advantage
| Feature | Traditional Biometrics | Brain-print |
|---------|-----------------------|-------------|
| Spoof Resistance | Low | High |
| Long-term Stability | Medium | High |
| Uniqueness | High | Very High |
| Acquisition Cost | Low | Medium |

## File Structure

```
.
├── Data/                          # MRI dataset (OASIS format)
│   └── Mild Dementia/             # Sample scans
├── mri_biometric.ipynb            # Main processing notebook
├── utils/
│   ├── preprocessing.py           # Image enhancement functions
│   └── visualization.py           # Plotting tools
├── Project Report.pdf             # Detailed technical report
└── README.md                      # This document
```

## References

1. Akkaynak, D., & Treibitz, T. (2019). *Sea-Thru: A Method for Removing Water From Underwater Images*. IEEE CVPR.
2. Aloui, K. et al. (2011). *A novel approach based brain biometrics*. IEEE CIBIM.
3. OASIS Dataset. *Open Access Series of Imaging Studies*.

## License

MIT License
