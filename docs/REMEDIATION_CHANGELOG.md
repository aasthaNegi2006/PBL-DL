# Remediation Changelog & Engineering Audit Trail

**Project:** Indian Sign Language (ISL) Emergency Gesture Recognition System  
**Academic Context:** UPES Semester V Deep Learning — Project-Based Evaluation (PBL-DL)  
**Branch:** `audit-pbe-readiness`  
**Date:** 21 November 2026  

---

## 1. Summary of Changes

| Issue ID | File Affected | Severity | Description of Defect | Remediation Applied |
|:---|:---|:---:|:---|:---|
| **P0-01** | `src/evaluate.py` | Critical | Hardcoded metrics (`test_accuracy=97.50`, `0.9778`, `78/80`, fixed signers `022-026`) in plotting function. | Refactored `plot_training_history` to strictly accept dynamic evaluation summary dicts; never assumes default numbers. |
| **P0-02** | `src/evaluate.py` | Critical | Missing experiment metadata in `summary.json`. | Added full authoritative schema: `checkpoint_sha256`, `dataset_fingerprint`, `num_samples`, `num_correct`, `macro_f1`, `weighted_f1`, `per_class_metrics`, `per_subject_accuracy`, `confusion_matrix`, and timestamp. |
| **P0-03** | `app/templates/index.html` | Critical | Static hardcoded table and badges showing 97.5% regardless of loaded model. | Bound cards, badges, and classification table dynamically to `/api/info`. Added cryptographic SHA-256 provenance check and warning banner for mismatched checkpoints. Sanitized hyperbolic language. |
| **P0-04** | `src/real_time_webcam.py` | Critical | False positive triggers on webcam; consecutive hits counter did not track candidate class. | Implemented candidate class tracking (`candidate_class`), requiring $N=3$ consecutive hits of the *exact same class*. Added hysteresis reset on class change, `none` prediction, or idle motion. Added 3-second speech cooldown. |
| **P0-05** | `src/dataset.py` | High | Unsafe `weights_only=False` fallback in `torch.load()`. | Removed unpickling fallback; strictly enforce `weights_only=True` for security against arbitrary code execution. |
| **P0-06** | `src/dataset.py` | High | Lack of explicit assertions on signer group disjointness. | Added defensive assertions ensuring $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, and $\text{val} \cap \text{test} = \emptyset$. Enforced regex validation on subject IDs. |
| **P0-07** | `src/data_prep.py` | High | Potential duplicate-recording leakage when `--include-uncropped` is used. | Added `recording_id` tracking to samples; added assertions preventing identical recordings from crossing split boundaries. |
| **P1-01** | `app/main.py` | High | `POST /api/predict` defaulted invalid video extensions (e.g. `.exe`) to `.avi` instead of rejecting. | Enforce strict extension validation (`.avi`, `.mp4`, `.mov`, `.mkv`, `.webm`) returning HTTP 415. Enforce 50 MB limit (HTTP 413) and ensure temporary files are deleted in `finally` blocks. |
| **P1-02** | `app/main.py` | High | `POST /api/predict-frames` lacked base64 validation and frame bounds checks. | Added base64 decoding error handling (HTTP 400), minimum frame count (8), and structured Pydantic response models. |
| **P1-03** | `run_web_dashboard.bat` | High | Defaulted uvicorn to `--host 0.0.0.0` (exposing video inference to local LAN). | Changed default to `--host 127.0.0.1`. Added pre-flight environment checks for virtualenv and checkpoint files. |
| **P1-04** | `run_video_demo.bat` | Medium | Unscientific claims ("guaranteed accurate detection"). | Updated copy to reflect offline video demonstration and added robust file existence checks. |
| **P1-05** | `tests/test_pipeline.py` | High | Incomplete test suite for security, provenance, and validation. | Added tests for HTTP 415 extension rejection, HTTP 413 size limits, SHA-256 provenance fields, duplicate-recording leakage rejection, and subject ID validation. |
| **P2-01** | `src/evaluate.py` | Low | Matplotlib warning on Windows sandbox non-writable directory. | Configured `MPLCONFIGDIR` to a local temporary directory to silence warnings and accelerate plot rendering. |

---

## 2. Test Suite Verification Summary

```text
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Aastha Negi\OneDrive - UPES\Documents\SignLanguageProject
collected 23 items

tests/test_pipeline.py::test_parse_subject_take PASSED                   [  4%]
tests/test_pipeline.py::test_subject_split_keeps_each_signer_in_one_split PASSED [  8%]
tests/test_pipeline.py::test_split_rejects_unknown_or_overlapping_subjects PASSED [ 13%]
tests/test_pipeline.py::test_dataset_eval_path_matches_live_path PASSED  [ 17%]
tests/test_pipeline.py::test_dataset_accepts_old_numpy_cache_and_augments PASSED [ 21%]
tests/test_pipeline.py::test_roi_box_stays_inside_frame[640-480] PASSED  [ 26%]
tests/test_pipeline.py::test_roi_box_stays_inside_frame[480-640] PASSED  [ 30%]
tests/test_pipeline.py::test_roi_box_stays_inside_frame[1920-1080] PASSED [ 34%]
tests/test_pipeline.py::test_roi_box_stays_inside_frame[112-112] PASSED  [ 39%]
tests/test_pipeline.py::test_frame_window_covers_a_whole_gesture PASSED  [ 43%]
tests/test_pipeline.py::test_model_forward_shape PASSED                  [ 47%]
tests/test_pipeline.py::test_checkpoint_loads_safely PASSED              [ 52%]
tests/test_pipeline.py::test_api_info PASSED                             [ 56%]
tests/test_pipeline.py::test_predict_frames_idle PASSED                  [ 60%]
tests/test_pipeline.py::test_predict_frames_moving PASSED                [ 65%]
tests/test_pipeline.py::test_predict_frames_rejects_bad_input PASSED     [ 69%]
tests/test_pipeline.py::test_predict_upload PASSED                       [ 73%]
tests/test_pipeline.py::test_unreadable_upload_is_rejected PASSED        [ 78%]
tests/test_pipeline.py::test_missing_model_returns_503 PASSED            [ 82%]
tests/test_pipeline.py::test_unsupported_upload_extension_returns_415 PASSED [ 86%]
tests/test_pipeline.py::test_api_info_provenance_fields PASSED           [ 91%]
tests/test_pipeline.py::test_duplicate_recording_leakage_rejected PASSED [ 95%]
tests/test_pipeline.py::test_subject_id_validation_rejects_malformed_signers PASSED [100%]

======================= 23 passed, 1 warning in 15.51s ========================
```
