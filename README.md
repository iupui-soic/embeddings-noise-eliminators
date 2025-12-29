
# Embeddings are Noise Eliminators

**Ablation studies to show what is eliminated in Foundation models**

## Overview

This repository contains the code and methodology for our study examining how foundation models handle synthetic noise in medical imaging. We demonstrate that Vision Transformer-based embeddings (RAD-DINO and DINOv3) act as intrinsic noise filters, suppressing irrelevant artifacts while preserving clinical information, in stark contrast to CNN architectures like ResNet50.

### Key Findings

- **Foundation models suppress synthetic noise**: RAD-DINO and DINOv3 showed poor detection of synthetic patterns (diagonal lines near random, AUC: 0.50–0.58) while maintaining strong disease classification (AUC: 0.71–0.91)
- **CNNs are highly sensitive to artifacts**: ResNet50 achieved near-perfect detection of all synthetic patterns (AUC: 0.93–0.99; F1: 0.90–0.99)
- **Domain-specific training matters**: RAD-DINO delivered superior clinical performance on NIH-CXR14 (cardiomegaly AUC: 0.91)
- **Implicit noise filtering**: Vision Transformer embeddings inherently filter out structured artifacts without explicit training


## Contact

For questions or collaboration opportunities, please contact:

**Saptarshi Purkayastha, PhD**  
Email: saptpurk@iu.edu  
Department of Biomedical Engineering and Informatics  
Indiana University, Indianapolis, IN, USA
