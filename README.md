## Brain-tumor-multi-class-classification-using-segmentation




A PyTorch-based deep learning pipeline for classifying brain MRI scans into four categories — **Glioma, Meningioma, No Tumor, Pituitary** — with two complementary classification pipelines, full explainability (Grad-CAM), uncertainty quantification (Monte Carlo Dropout), robustness testing (Test-Time Augmentation), and automated clinical PDF reporting.

---

## 📋 Overview

| | |
|---|---|
| **Backbone** | EfficientNetB0 (ImageNet pre-trained) |
| **Task** | 4-class classification: Glioma · Meningioma · No Tumor · Pituitary |
| **Dataset** | BRISC2025 (`briscdataset/brisc2025` via kagglehub) |
| **Segmentation** | Attention U-Net (proxy masks via CLAHE + Otsu) |
| **Explainability** | Grad-CAM (last backbone Conv2d layer) |
| **Uncertainty** | Monte Carlo Dropout (30 forward passes) |
| **Robustness** | Test-Time Augmentation (10 augmentations, weighted ensemble) |
| **Reporting** | JSON + PDF (ReportLab, A4 clinical format) |
| **Environment** | Tested on Kaggle T4/P100 and Google Colab T4; also run on NVIDIA GB10 (ARM) |

> Run all cells top-to-bottom. Enable GPU: `Runtime → Change runtime type → T4 GPU` (or equivalent).

---

## 🏗️ Architecture: Two Classification Pipelines

This notebook builds **two separate classifiers** so their performance can be directly compared on the same sealed test set.

### Pipeline 1 — Standard Classifier
```
crop_brain_contour() → NeuroScanNet_v2 (frozen EfficientNetB0 backbone) → Softmax(4)
```
- Trains only the classification head (329K / 4.3M params trainable)
- No knowledge of tumor location — classifies the whole brain slice

### Pipeline 2 — ROI-Guided Classifier
```
crop_brain_contour() → Attention U-Net (segmentation) → extract_roi_from_mask() 
                     → NeuroScanNet_v2 (progressively fine-tuned) → Softmax(4)
```
- A separate Attention U-Net first segments the tumor region
- The classifier trains and predicts on **cropped tumor ROIs**, not full brain frames
- Backbone is progressively unfrozen across 3 fine-tuning stages

Both use the identical `NeuroScanNet_v2` architecture:
```
EfficientNetB0 → GlobalAvgPool → Dropout(0.40) → Dense(256, ReLU) → BatchNorm 
              → Dropout(0.30) → Linear(4)
```

---

## 📊 Results Summary

| Metric | Standard Classifier | ROI-Guided Classifier |
|---|---|---|
| **Test Accuracy** | 93.69% | **94.77%** |
| **Macro F1** | 93.83% | **94.86%** |
| **Macro Precision** | 93.87% | 94.89% |
| **Macro Recall** | 93.82% | 94.84% |
| Train/Val Gap | −4.29% (healthy) | +3.84% (mild overfitting) |
| Val/Train Loss Ratio | 0.71× | 3.74× (moderate divergence) |

### Per-Class Recall Comparison

| Class | Standard | ROI-Guided | Δ |
|---|---|---|---|
| Glioma | 90.7% | 91.3% | +0.6% |
| **Meningioma** | **87.8%** | **92.4%** | **+4.6%** |
| No Tumor | 100.0% | 99.4% | −0.6% |
| Pituitary | 96.8% | 96.3% | −0.5% |

**Key finding:** The ROI-guided pipeline's overall accuracy gain is driven almost entirely by improved meningioma detection — small, off-center lesions benefit most from tumor-focused cropping. Other classes are flat or marginally unchanged, and the ROI pipeline shows measurably more overfitting risk than the standard pipeline.

---

## 🔬 Pipeline Details

### 1. Data Preparation
- **Dataset**: BRISC2025, 5,000 images across 4 classes
- **Deduplication**: MD5 hashing removes exact duplicates *before* splitting (prevents train/test leakage) → 4,962 unique images
- **Split**: Stratified 70/15/15 (train/val/test), with explicit leakage-guard assertions
- **Class balancing**: `compute_class_weight(balanced)` applied during training to counter class imbalance (no_tumor is the smallest class)
- **Preprocessing**: Otsu-threshold brain contour cropping → 224×224, float32 in [0, 255] (EfficientNet normalizes internally — do **not** divide by 255 upstream)

### 2. Segmentation (Attention U-Net)
- Full 5-level encoder / 4-level decoder with attention gates on every skip connection (~8.07M params)
- **No ground-truth masks exist** for this dataset — proxy masks are generated via CLAHE + Otsu threshold + morphological cleanup + largest-connected-component extraction
- Trained with combined BCE + Dice loss, 50 epochs max
- **Result**: Best val Dice = 0.8554, val IoU = 0.7498
- ⚠️ Known limitation: since supervision is heuristic (not radiologist-verified), the segmenter occasionally fails on low-contrast cases (observed: one glioma sample with IoU = 0.000)

### 3. ROI Extraction
- `extract_roi_from_mask()` crops to the largest contour of the predicted mask, with 15% padding
- **Fallback logic**: if the detected region is smaller than 10% of image dimensions (segmentation failure), the full preprocessed image is used instead — this protects the classifier from garbage crops but means some "ROI-guided" samples are effectively standard crops in disguise

### 4. Progressive Fine-Tuning (Stage A → B → C)
| Stage | Backbone State | Epochs | LR Schedule |
|---|---|---|---|
| A | Frozen (head only) | 15 | Cosine decay from 1e-3 |
| B | Top 20 non-BN layers unfrozen | 20 | Cosine decay from 1e-4 + ReduceLROnPlateau |
| C | Top 50 non-BN layers unfrozen | 25 | Cosine decay from 5e-5 + ReduceLROnPlateau |

BatchNorm layers remain frozen throughout to preserve ImageNet statistics.

### 5. Explainability & Uncertainty
- **Grad-CAM**: Forward/backward hooks on the last backbone Conv2d layer, applied to both standard and ROI-guided predictions
- **Monte Carlo Dropout**: 30 stochastic forward passes (dropout active, BatchNorm frozen) → mean prediction + per-class uncertainty (σ)
- **Test-Time Augmentation**: 10 randomly augmented forward passes, averaged
- **Ensemble**: Final production inference blends Standard (50%) + MC Dropout (30%) + TTA (20%)

### 6. Reporting
- 9-panel production dashboards (accuracy/loss curves, confusion matrix, class distributions, generalization gap, per-class accuracy)
- Automated model audit (overfitting detection, weak-class flagging, loss divergence checks)
- Hospital-style A4 PDF clinical report (ReportLab) with risk stratification, probability breakdown, and recommendations

---

## 📁 Output Artifacts

| File | Description |
|---|---|
| `final_neuroscan_model.pt` | Standard classifier (final) |
| `best_brain_tumor_model.pt` | Standard classifier (best checkpoint) |
| `best_seg_model.pt` | Attention U-Net segmenter |
| `final_roi_classifier.pt` | ROI-guided classifier (final) |
| `stage_a_best.pt` / `stage_b_best.pt` / `stage_c_best.pt` | Per-stage fine-tuning checkpoints |
| `learning_curves.png` | Standard classifier training history |
| `seg_training_curves.png` | Segmenter training history |
| `confusion_matrix_roc.png` | Standard classifier evaluation |
| `phase3_finetuning_history.png` | Stage A/B/C merged training history |
| `gradcam_gallery.png` | Grad-CAM examples per class |
| `unified_xai_*.png` | 5-panel explainability figures per class |
| `neuroscan_dashboard.png` / `neuroscan_dashboard_roi.png` | Production dashboards |
| `NeuroScan_Report.pdf` | Clinical PDF report |
| `seg_training_log.csv` | Per-epoch segmentation metrics |

---

## ⚙️ Setup & Usage

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python-headless scikit-learn matplotlib seaborn pandas scipy \
            reportlab streamlit tqdm kagglehub psutil Pillow watchdog timm
```

Run the notebook top-to-bottom. It auto-detects Kaggle vs. Colab/local environments for dataset mounting.

### Optional: Live Web Demo
The final cell launches a Streamlit app with a public Cloudflare tunnel URL (no account required):
```
Run the "One-Click Streamlit Launcher" cell after uploading app.py
```
⚠️ Note: the deployment snippet in this notebook references TensorFlow's `.keras` format in one section — this is stale from a pre-PyTorch version and should be updated to `torch.load(...)` before deploying `app.py`.

---

## ⚠️ Known Limitations

1. **Proxy segmentation masks**: The Attention U-Net is trained against Otsu-threshold heuristics, not radiologist-verified ground truth. Its Dice/IoU scores measure agreement with this heuristic, not true clinical accuracy.
2. **ROI fallback ambiguity**: When segmentation confidence is low, the pipeline silently substitutes the full image — the exact fraction of test samples affected by this fallback is not tracked or reported.
3. **ROI classifier overfitting**: Stage C fine-tuning shows a widening train/val gap (3.84%) and rising val_loss despite falling train_loss — a sign of mild overfitting not present in the standard classifier.
4. **Stale hardcoded values**: The PDF report generator and Phase 5 deployment snippet contain hardcoded accuracy figures and TF/Keras references left over from earlier notebook iterations — these should be parameterized before production use.
5. **Research tool only**: This system is explicitly **not a medical diagnostic device**. All outputs require clinical correlation and expert radiological review.

---

## 🏥 Clinical Disclaimer

> NeuroScan AI is an experimental research system. Reports generated by this pipeline are **NOT** medical diagnoses and have **NOT** been reviewed by a licensed clinician. All findings must be correlated with clinical history and expert radiological interpretation before any clinical decision. The developers assume no liability for clinical decisions based on this output.

---

## 📚 Class Reference

| Class | Description |
|---|---|
| **Glioma** | Primary brain tumor arising from glial cells (WHO Grade I–IV) |
| **Meningioma** | Typically slow-growing extra-axial tumor arising from arachnoid cap cells (WHO Grade I–III) |
| **No Tumor** | No imaging features consistent with intracranial neoplasm |
| **Pituitary** | Lesion in the pituitary region; warrants endocrinological evaluation |

---
