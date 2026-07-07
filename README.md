# Brain-tumor-multi-class-classification-using-segmentation

## End-to-end execution flow

```text
Cells 1-11  (env, data, split, weights, generators)
        │
        ▼
Cells 12-20 (Original classifier: build → train → eval → save)
        │                                   │
        │                                   ▼
        │                          Cells 21-22 (MC Dropout, TTA on Original model)
        │                                   │
        │                                   ▼
        │                          run_end_to_end() → PDF report
        │
        ▼
Phase 1 (Attention U-Net: independent training on proxy masks)
        │
        ▼
Phase 2 (ROI extraction using Phase-1's seg_model, with fallback)
        │
        ▼
Phase 3 (roi_clf: Stage A→B→C fine-tuning on ROI-cropped images)
        │
        ▼
Phase 4 (Unified explainability: seg_model + roi_clf + Grad-CAM combined)
        │
        ▼
Cell 25 dashboards (both models) → Audit cell (compares both) → Final summary
```