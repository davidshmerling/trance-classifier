# 🎧 TranceClassifier

TranceClassifier is a machine-learning project designed to classify trance tracks into three sub-genres: **Goa**, **Psytrance**, and **Dark**.

The system extracts audio features (MFCC, spectral features, chroma, tempo), builds spectrogram embeddings, and trains deep-learning models (CNN / GRU / CRNN / Transformer).  
Each training run automatically generates a new version folder with analysis outputs.

---

## 🚀 Features
- Audio feature extraction & preprocessing  
- Support for Embedding input, Meta-features, or both  
- Multiple model architectures (CNN, GRU, CRNN, Transformer)  
- Automatic versioning (`v1`, `v2`, `v3`, …)  
- Confusion matrix, accuracy/loss plots, and classification report  
- Gradient-based input importance  
- Easy configuration through `config.py`

---

## 🎯 Model Performance
Typical model accuracy: **58%–63%**,  
significantly higher than the random baseline of 33% (3 classes).  
Trance sub-genres overlap heavily, so even 60%+ is meaningful and non-trivial.

---






## 👤 Author

David Shmerling  
*(Add your image below if you want)*

![Profile Image](YOUR_IMAGE_HERE.jpg)