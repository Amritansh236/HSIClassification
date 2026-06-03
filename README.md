# 🛰️ ViT-Inspired Classifier for Hyperspectral Image (HSI) Classification

This project implements a **lightweight Vision Transformer (ViT)-inspired model** using **PyTorch** to perform patch-based classification on the **Indian Pines hyperspectral dataset**.  

The script loads HSI data, extracts **7×7 patches** around labeled pixels, trains a transformer-based classifier, and generates a final prediction map, which is then compared against the ground truth.

---

## ✨ Features

- **Data Loading:** Loads `.mat` files for HSI data (`Indian_pines_corrected.mat`) and ground truth (`Indian_pines.mat`).  
- **Normalization:** Applies per-band z-score normalization to the HSI cube.  
- **Patch Extraction:** Efficiently extracts 7×7 patches for labeled pixels only.  
- **Class Filtering:** Handles class imbalance by removing classes with fewer than 10 samples, ensuring a stable train/test split.  
- **Model:** Uses a custom `SimpleViTClassifier` built with `torch.nn.TransformerEncoder`.  
- **GPU Acceleration:** Automatically utilizes a CUDA-enabled GPU if available.  
- **Training:** Implements a standard loop with mini-batching, Adam optimizer, and CrossEntropyLoss.  
- **Evaluation:**
  - Calculates test accuracy on a held-out set.  
  - Computes pixel-level accuracy across the entire map.  
- **Visualization:** Generates side-by-side comparisons of:
  - Ground Truth  
  - Model Predictions  
  - Prediction Errors  

---

## 🗺️ Dataset

This project uses the **Indian Pines** hyperspectral dataset.
You can obtain these from various academic sources, such as the **Purdue University Research Repository (PURR)**.


## 📜 License

This project is licensed under the **MIT License**.
