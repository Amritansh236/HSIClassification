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

This project uses the **Indian Pines** hyperspectral dataset. You’ll need two files:

- `Indian_pines_corrected.mat` → The corrected HSI data cube (145×145×200)  
- `Indian_pines.mat` → Ground truth labels (145×145)

You can obtain these from various academic sources, such as the **Purdue University Research Repository (PURR)**.

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   Make sure you have a `requirements.txt` with the necessary packages, then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   Place both `.mat` files in the project root (or update paths in the script accordingly).

---

## 🧩 Usage

Before running, update the data file paths at the top of the script:

```python
# --- UPDATE THESE PATHS ---
img_path = "path/to/Indian_pines_corrected.mat"
gt_path = "path/to/Indian_pines.mat"

img = loadmat(img_path)['indian_pines_corrected']
gt_raw = loadmat(gt_path)['indian_pines']
# -------------------------
```

Then, run the classifier:

```bash
python hsi_vit_classifier.py
```

The script will:

* Display which device is being used (CPU or GPU).
* Load, normalize, and extract patches.
* Show class distribution and filtered classes.
* Print training progress (loss & accuracy) every 5 epochs.
* Report **final Test Accuracy**.
* Save model weights as `trained_model.pth`.
* Generate and display classification maps via matplotlib.
* Print pixel-level accuracy and error stats to console.

---

## 🧠 Model Architecture

The `SimpleViTClassifier` is a **minimal transformer-based model**. Instead of generating multiple spatial patches like a traditional ViT, it treats each **7×7×200** patch as a single token.

* **Input:** Flattened patch `[batch_size, 9800]` (since 7×7×200 = 9800).
* **Patch Embedding:** Linear projection → `embed_dim` (e.g., 256).
* **Positional Embedding:** Learnable `pos_embedding` added to the token.
* **Transformer Encoder:** 4-layer `nn.TransformerEncoder`.
* **Classifier Head:** Output passed through a final linear layer → logits for `num_classes`.

---

## 📜 License

This project is licensed under the **MIT License**.

---

Would you like me to make a small badge header (Python / PyTorch / ViT / MIT License) for this one too, to match the aesthetic of your Spotify dashboard README?
```
