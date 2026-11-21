# Baseline Audit & Engineering Assessment: PBL-DL

**Project:** Indian Sign Language (ISL) Emergency Gesture Recognition System  
**Academic Context:** UPES Semester V Deep Learning — Project-Based Evaluation (PBE)  
**Evaluation Date:** Monday, 23 November 2026 (Strictly Individual / Solo Viva)  
**Evaluator Guidelines:** 8 minutes live demonstration + 4 minutes technical viva voce  
**Audit Date:** 21 November 2026  
**Git Branch:** `audit-pbe-readiness` (branched from `main` at commit `6dc8795`)  
**Audit Author:** Deep Learning Engineering & Academic Verification Team  

---

## 1. Executive Summary

This audit assesses the reproducibility, scientific integrity, runtime stability, and academic defensibility of the **ISL Emergency Gesture Recognition System** prior to final evaluation.

The system is designed to recognize five emergency gestures (`accident`, `call`, `doctor`, `help`, `pain`) alongside a negative rejection class (`none`) using a spatio-temporal deep neural network:
- **Spatial Feature Extractor:** 4-stage Residual 2D-CNN ($112 \times 112 \times 3 \to 256$-dim embedding per frame)
- **Temporal Sequence Modeling:** 2-layer Bidirectional Gated Recurrent Unit (BiGRU, hidden dimension 128)
- **Temporal Saliency:** Bahdanau-style Additive Attention mechanism aggregating 16 frames into a 256-dim context vector
- **Classification Head:** Linear layer projecting to logits for 6 target classes

The system was evaluated on a held-out test partition of 5 completely unseen signers (`022`–`026`, 80 total video clips), yielding a reported test accuracy of **97.50%** (78/80 correct) and **0.9778 Macro F1**, comfortably exceeding the academic target threshold ($\ge 85.0\%$).

However, our baseline audit identified several critical engineering defects, security vulnerabilities, and presentation discrepancies that require immediate remediation before viva defense.

---

## 2. Environment & Artifact Baseline

| Component | State / Version | Verification Status |
|:---|:---|:---|
| **OS** | Windows 11 AMD64 | Confirmed |
| **Python** | 3.13.1 (64-bit) | Confirmed |
| **PyTorch** | 2.15.0+cpu | Verified (CPU inference target) |
| **Torchvision** | 0.29.2 | Confirmed |
| **OpenCV** | 5.0.0.93 | Confirmed |
| **FastAPI / Starlette** | 0.162.7 / 1.10.0 | Confirmed |
| **Pydantic** | 2.14.3 | Confirmed |
| **Model Weights** | `models/best_isl_model.pth` | Confirmed (7.7 MB, SHA-256: `52781568...`) |
| **Cached Dataset** | `data/processed/cached_all_frames.pt` | Confirmed (1,145.7 MB, 412 clips, 26 signers) |
| **Test Suite** | `tests/test_pipeline.py` | 19 / 19 tests passing (12.5s) |

---

## 3. Academic Evaluation Criteria & Viva Alignment

Analysis of the official course documents (`DL_Sem5_Evaluation_Guidelines_23_Nov_2026.pdf`, `DL_Sem5_Evaluation_FAQs_23_Nov_2026.pdf`, and `Aastha_Task_Guide.pdf`) reveals the key expectations of the evaluation panel:

1. **Solo Accountability:** The student must independently navigate code, explain all math, and justify design decisions.
2. **Beyond Accuracy:** High accuracy alone is insufficient. The panel expects explanation of dataset provenance, split design, leakage prevention, loss minimization, backpropagation, and error analysis.
3. **Leakage Defense:** Explaining why the earlier 100% accuracy was an artifact of identity leakage (same signer in train and test takes) and how strict signer-independent group splitting resolved it.
4. **Authenticity & Integrity:** Zero fabrication of results; clear acknowledgment of libraries, open datasets, and AI assistance.
5. **Practical Generalization vs. Studio Artifacts:** Understanding domain shift—the difference between studio-recorded clips (dark blue backdrop, uniform distance) and ambient live webcam input (illumination variations, sensor noise, background clutter).

---

## 4. Classified Audit Findings

### Priority P0: Critical (Scientific Integrity & Core Logic)

- **[DEFECT-01] Hardcoded Evaluation Metrics in Plotting Function (`src/evaluate.py`):**
  - *Finding:* `plot_training_history()` embedded hardcoded references to `test_accuracy=97.50`, `78/80`, `0.9778`, and fixed signer IDs (`022–026`). Calling this function on a newly trained model or fold would produce misleading graphics.
  - *Fix:* Refactor plotting to strictly accept a verified evaluation summary dictionary containing dynamic metrics, signer lists, and sample counts.

- **[DEFECT-02] Static Hardcoded Performance Claims on Web Dashboard (`app/templates/index.html`):**
  - *Finding:* Metric cards and classification report table contained static HTML values (`97.50%`, `0.9778`, `78/80`). Furthermore, there was no provenance validation verifying that the evaluation artifact belongs to the loaded checkpoint.
  - *Fix:* Bind dashboard cards and tables dynamically to `/api/info`. Validate checkpoint SHA-256 and model version against the evaluation summary; show a provenance warning if unverified or missing. Remove hyperbolic phrases ("guaranteed", "fraudulent").

- **[DEFECT-03] State Machine Flaw in Real-Time Desktop Recognizer (`src/real_time_webcam.py`):**
  - *Finding:* `consecutive_hits` accumulated whenever `top_conf >= CONFIDENCE_THRESHOLD` without verifying that the predicted class matched the preceding candidate class. A rapid alternating sequence (e.g. `HELP` $\to$ `ACCIDENT`) could falsely trigger confirmation.
  - *Fix:* Implement explicit candidate tracking (`candidate_class`), requiring $N$ consecutive frames of the exact same class. Reset counter on class change, `none` prediction, or motion energy dropping below idle threshold.

- **[DEFECT-04] Signer-Independent Group Disjointness Enforcement:**
  - *Finding:* While `src/dataset.py` implemented `split_by_subject()`, explicit runtime assertions verifying `train ∩ val == ∅`, `train ∩ test == ∅`, and `val ∩ test == ∅` were missing across some split paths.
  - *Fix:* Add defensive validation and automated regression tests ensuring disjoint partitions across all training and evaluation entry points.

---

### Priority P1: High (Security, Reliability & Error Handling)

- **[DEFECT-05] Arbitrary Code Execution Risk via Unsafe Pickle Fallback (`src/dataset.py`):**
  - *Finding:* `load_cache()` fell back to `torch.load(path, weights_only=False)` upon unpickling errors, exposing the system to potential malicious payload execution.
  - *Fix:* Remove `weights_only=False` fallback. Require all caches to be in safe PyTorch tensor format.

- **[DEFECT-06] Inadequate Input Validation on Video Upload (`app/main.py`):**
  - *Finding:* `POST /api/predict` defaulted invalid video extensions (e.g. `.exe`) to `.avi` rather than returning HTTP 415 (Unsupported Media Type).
  - *Fix:* Enforce strict extension validation (`.avi`, `.mp4`, `.mov`, `.mkv`, `.webm`) returning HTTP 415, enforce 50 MB limit (HTTP 413), and ensure temporary files are deleted in a `finally` block.

- **[DEFECT-07] Network Exposure & Launcher Script Fragility:**
  - *Finding:* `run_web_dashboard.bat` bound uvicorn to `--host 0.0.0.0`, exposing the camera inference server to the local network. Batch scripts lacked existence checks for the Python interpreter and model weights.
  - *Fix:* Change default binding to `--host 127.0.0.1`. Add robust existence checks and clear diagnostic error messages to all `.bat` launcher scripts.

---

### Priority P2: Medium (Usability, Performance & Test Harness)

- **[DEFECT-08] Windows Test Harness Compatibility:**
  - *Finding:* Starlette's `TestClient` uses AnyIO blocking portals that require local loopback socket binding. In isolated sandbox environments, this can hang.
  - *Fix:* Document execution requirements and implement lightweight direct ASGI tests or configure proper transport flags.

- **[DEFECT-09] Matplotlib Config Directory Warning:**
  - *Finding:* Matplotlib logs warnings regarding non-writable `~/.matplotlib` on Windows sandbox runs, rebuilding font cache repeatedly.
  - *Fix:* Set `MPLCONFIGDIR` to a local workspace temporary directory in plotting scripts.

---

### Priority P3: Low (Documentation & Polish)

- **[DEFECT-10] Documentation Completeness:**
  - *Finding:* Need formal academic documentation covering architecture, model card, evaluation protocols, viva defense questions, and remediation changelog.
  - *Fix:* Compile comprehensive documentation in `docs/` and update `README.md`.

---

## 5. Remediation Plan & Roadmap

1. **Phase 1:** Refactor `src/evaluate.py` to eliminate hardcoded metrics and standardize experiment results schema.
2. **Phase 2:** Harden `src/dataset.py` and `src/train.py` with disjoint group assertions and safe tensor loading.
3. **Phase 3:** Overhaul the real-time confirmation state machine in `src/real_time_webcam.py`.
4. **Phase 4:** Harden FastAPI endpoints in `app/main.py` (HTTP status codes, Pydantic schemas, cleanup).
5. **Phase 5:** Dynamically bind `app/templates/index.html` to backend metrics with provenance checks.
6. **Phase 6:** Secure launcher scripts (`.bat`) and sanitize claims.
7. **Phase 7:** Expand automated test suite in `tests/` covering dataset splitting, state machine, and API validation.
8. **Phase 8:** Complete comprehensive academic documentation suite (`docs/`).
