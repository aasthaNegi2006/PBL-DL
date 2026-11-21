# Model Card: ISL Emergency Gesture Recognizer

**Model Name:** `isl-spatiotemporal-resnet-bigru-att-v1`  
**Checkpoint Path:** `models/best_isl_model.pth`  
**Checkpoint Digest (SHA-256):** `527815680e36c5b44fe135f3cbee3c018ad00aaf04caae98af842a92af03ec09`  
**Release Date:** November 2026  
**License:** MIT (Academic Evaluation & Research Prototype)  
**Framework:** PyTorch 2.15.0+cpu  

---

## 1. Intended Use

### Primary Intended Use
- Real-time gesture recognition of 5 high-priority emergency words in Indian Sign Language (ISL):
  1. `accident`
  2. `call`
  3. `doctor`
  4. `help`
  5. `pain`
  Alongside an out-of-vocabulary negative rejection class: `none`.
- Academic evaluation for UPES Fifth-Semester Deep Learning Project-Based Learning (PBL-DL).
- Assistive emergency translation prototype for Deaf and Hard-of-Hearing (DHH) individuals.

### Out-of-Scope & Prohibited Uses
- **Life-Safety Critical Dispatch:** This system is an academic research prototype. It must not be deployed as an unassisted, standalone emergency dispatcher in real 911/112 dispatch systems without human verification.
- **Continuous Sentence Translation:** The model recognizes isolated emergency signs; it does not translate continuous ISL grammatical discourse, non-manual facial markers, or regional dialect variations.

---

## 2. Model Architecture & Training Factors

- **Input Dimension:** Tensor of shape `(Batch, 16, 3, 112, 112)` normalized via ImageNet mean and standard deviation.
- **Spatial Encoder:** 4-stage Residual 2D-CNN ($3 \to 32 \to 64 \to 128 \to 256$) with residual skip connections and global average pooling.
- **Temporal Sequence Model:** 2-layer Bidirectional GRU (Hidden dimension = 128 per direction, total = 256).
- **Attention Mechanism:** Bahdanau-style Additive Attention producing a single 256-dimensional summary context vector.
- **Classification Head:** Linear layer mapping from 256 dimensions to 6 class logits.
- **Parameter Count:** 1,918,247 parameters (7.7 MB FP32 checkpoint).

### Training Hyperparameters
- **Epochs:** 40 (Best model selected at Epoch 34 based on validation loss)
- **Batch Size:** 8
- **Base Learning Rate:** $3 \times 10^{-4}$ with AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$)
- **Learning Rate Schedule:** Cosine Annealing learning rate schedule decay
- **Augmentation:** Random temporal jitter sampling, spatial translations ($\pm 6\text{ px}$), contrast jitter ($\pm 15\%$), brightness scaling ($\pm 15\%$)

---

## 3. Data & Evaluation Protocol

- **Dataset Source:** "A Video Dataset of the Hand Gestures of Indian Sign Language Words used in Emergency Situations", Mendeley Data (CC BY 4.0).
- **Total Dataset Size:** 412 video clips across 26 distinct signers (260 emergency clips, 152 negative `none` clips).
- **Disjoint Partitioning Protocol (Signer-Independent Group Split):**
  - **Training Signers (17 signers, 268 clips):** `['002', '003', '005', '006', '007', '008', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020']`
  - **Validation Signers (4 signers, 64 clips):** `['001', '004', '009', '021']`
  - **Held-Out Test Signers (5 signers, 80 clips):** `['022', '023', '024', '025', '026']`
- **Data Leakage Guarantee:**
  $$\text{Signers}_{\text{train}} \cap \text{Signers}_{\text{val}} = \emptyset, \quad \text{Signers}_{\text{train}} \cap \text{Signers}_{\text{test}} = \emptyset, \quad \text{Signers}_{\text{val}} \cap \text{Signers}_{\text{test}} = \emptyset$$

---

## 4. Quantitative Performance Metrics

### Official Held-Out Test Evaluation (Signers 022–026, 80 Clips)

| Metric | Measured Value | Academic Target Threshold | Status |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **97.50%** (78 / 80) | $\ge 85.0\%$ | **PASSED (+12.5% margin)** |
| **Macro-Averaged F1** | **0.9778** | — | **Robust** |
| **Weighted-Averaged F1** | **0.9750** | — | **Robust** |

### Per-Class Detailed Performance

| Class | Type | Precision | Recall | F1-Score | Support |
|:---|:---|:---:|:---:|:---:|:---:|
| `accident` | Emergency | 1.0000 | 1.0000 | 1.0000 | 10 |
| `call` | Emergency | 0.9000 | 0.9000 | 0.9000 | 10 |
| `doctor` | Emergency | 1.0000 | 1.0000 | 1.0000 | 10 |
| `help` | Emergency | 1.0000 | 1.0000 | 1.0000 | 10 |
| `pain` | Emergency | 1.0000 | 1.0000 | 1.0000 | 10 |
| `none` | Negative Rejection | 0.9667 | 0.9667 | 0.9667 | 30 |

### Per-Signer Test Breakdown
- **Signer 022:** 100.0% (16 / 16 correct)
- **Signer 023:** 100.0% (16 / 16 correct)
- **Signer 024:** 87.5% (14 / 16 correct — 1 `call` misclassified into `none`)
- **Signer 025:** 100.0% (16 / 16 correct)
- **Signer 026:** 100.0% (16 / 16 correct)

---

## 5. Domain Shift & Technical Limitations

1. **Background & Lighting Shift:**
   The training videos were recorded in a controlled studio setting with a dark-blue fabric backdrop and static illumination. When tested in ambient room settings, resting background motion can cause false activations if unconstrained. This is mitigated through:
   - **Calibrated Motion Energy Gating ($\theta_{\text{motion}} = 2.4$):** Automatically marks sub-threshold motion as `idle` without calling the deep ConvNet.
   - **Torso-Focused ROI Framing ($c_x = 0.50, c_y = 0.58, \text{size} = 0.75$):** Emphasizes hands and chest while excluding facial glare.
   - **Confidence Margin Gating ($\Delta_{\text{margin}} \ge 0.12$):** Rejects ambiguous predictions where the top logit fails to dominate runner-up candidates.

2. **Temporal Windowing:**
   The model expects a complete gesture executed over 2.0–3.0 seconds. Fast, jerky or incomplete motions will be suppressed until confirmed across consecutive window hits.
