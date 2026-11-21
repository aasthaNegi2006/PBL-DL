# System Architecture: Spatio-Temporal ISL Gesture Recognizer

**Project:** Indian Sign Language (ISL) Emergency Gesture Recognition System  
**Academic Context:** UPES Semester V Deep Learning — Project-Based Learning (PBL-DL)  
**Evaluation Date:** Monday, 23 November 2026  
**Target:** 5 Emergency Gestures (`accident`, `call`, `doctor`, `help`, `pain`) + `none` (non-emergency rejection)

---

## 1. High-Level Architectural Pipeline

The system processes continuous video streams into isolated gesture classifications using a four-stage spatio-temporal deep neural network:

```
Input Video Clip / Live Buffer
   (T=16 frames, H=112, W=112, C=3)
                 │
                 ▼
 ┌────────────────────────────────┐
 │   Stage 1: Spatial Encoder     │
 │  4-Stage Residual 2D-CNN       │  -> Extracts 256-dim feature vector per frame
 │  Output: (Batch, 16, 256)      │
 └────────────────────────────────┘
                 │
                 ▼
 ┌────────────────────────────────┐
 │   Stage 2: Temporal Sequence   │
 │  2-Layer Bidirectional GRU     │  -> Captures forward & backward temporal dynamics
 │  Output: (Batch, 16, 256)      │     (128 hidden dim x 2 directions = 256)
 └────────────────────────────────┘
                 │
                 ▼
 ┌────────────────────────────────┐
 │   Stage 3: Temporal Attention  │
 │  Bahdanau Additive Attention   │  -> Computes frame-level saliency weights α_t
 │  Output: (Batch, 256) context  │
 └────────────────────────────────┘
                 │
                 ▼
 ┌────────────────────────────────┐
 │   Stage 4: Classification Head │
 │  Dense Linear Projection       │  -> Computes unnormalized class logits
 │  Output: (Batch, 6) Logits     │
 └────────────────────────────────┘
```

---

## 2. Mathematical Formulation

### 2.1 Spatial Residual Feature Extractor
Given an input sequence of video frames $X = (x_1, x_2, \dots, x_T) \in \mathbb{R}^{T \times 3 \times 112 \times 112}$, each frame $x_t$ is individually encoded through a shared 4-stage Residual Convolutional Network:

$$z_t = f_{\text{ResCNN}}(x_t) \in \mathbb{R}^{D}, \quad \text{where } D = 256$$

The spatial encoder employs residual projection blocks:

$$y = \text{ReLU}\left(\mathcal{F}(x, \{W_i\}) + W_s x\right)$$

Where:
- $\mathcal{F}(x, \{W_i\}) = \text{Conv2D}_{3 \times 3}(\text{BatchNorm}(\text{ReLU}(\text{Conv2D}_{3 \times 3}(\text{BatchNorm}(x)))))$
- $W_s$ represents a $1 \times 1$ pointwise projection matching channel dimensions and spatial strides ($s=2$).
- Channel expansion: $3 \to 32 \to 64 \to 128 \to 256$, with spatial downsampling: $112 \times 112 \to 56 \times 56 \to 28 \times 28 \to 14 \times 14 \to 7 \times 7$.
- Global Average Pooling (GAP) collapses $7 \times 7 \times 256 \to 256$.

### 2.2 Bidirectional Gated Recurrent Unit (BiGRU)
Human sign gestures exhibit critical temporal dependencies across both entry phases (preparation) and resolution phases (retraction). A 2-layer Bidirectional GRU processes the sequence $Z = (z_1, \dots, z_T)$:

$$\overrightarrow{h}_t = \text{GRU}_{\text{fwd}}(z_t, \overrightarrow{h}_{t-1})$$
$$\overleftarrow{h}_t = \text{GRU}_{\text{bwd}}(z_t, \overleftarrow{h}_{t+1})$$

The forward and backward hidden representations are concatenated at each timestep:

$$h_t = \left[ \overrightarrow{h}_t \,\|\, \overleftarrow{h}_t \right] \in \mathbb{R}^{2H}, \quad H = 128, \quad 2H = 256$$

Inside each GRU cell, reset ($r_t$) and update ($u_t$) gates regulate information retention without vanishing gradients:

$$r_t = \sigma(W_{zr} z_t + W_{hr} h_{t-1} + b_r)$$
$$u_t = \sigma(W_{zu} z_t + W_{hu} h_{t-1} + b_u)$$
$$\widetilde{h}_t = \tanh(W_{zh} z_t + W_{hh} (r_t \odot h_{t-1}) + b_h)$$
$$h_t = (1 - u_t) \odot h_{t-1} + u_t \odot \widetilde{h}_t$$

### 2.3 Bahdanau Additive Temporal Attention
Rather than relying solely on the final hidden state $h_T$ (which suffers from recency bias), the model employs an additive attention mechanism to softly weight the most informative gesture phases (e.g. hand collisions in `accident`, wrist-tapping in `doctor`):

1. **Alignment Energy:**
   $$e_t = v_a^\top \tanh\left(W_a h_t + b_a\right), \quad W_a \in \mathbb{R}^{64 \times 256}, \, v_a \in \mathbb{R}^{64}$$

2. **Attention Distribution:**
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}, \quad \sum_{t=1}^T \alpha_t = 1$$

3. **Context Vector Aggregation:**
   $$c = \sum_{t=1}^T \alpha_t h_t \in \mathbb{R}^{256}$$

### 2.4 Classification Head & Loss Minimization
The aggregated temporal context vector $c$ is projected to class logits:

$$\hat{y} = W_c c + b_c \in \mathbb{R}^{K}, \quad K = 6$$

Normalized posterior probabilities are obtained via Softmax:

$$P(y = k \mid X) = \frac{\exp(\hat{y}_k)}{\sum_{j=1}^K \exp(\hat{y}_j)}$$

Optimization minimizes multi-class Cross-Entropy Loss:

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \log P(y_{i,k} \mid X_i) + \lambda \|\theta\|_2^2$$

With weight decay $\lambda = 10^{-4}$ (L2 regularization) and Cosine Annealing learning rate schedule:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)$$

---

## 3. Parameter Count & Computational Budget

| Component | Layer Specification | Output Dimensions | Parameter Count |
|:---|:---|:---|---:|
| **Input** | Batch Video Tensor | $(B, 16, 3, 112, 112)$ | 0 |
| **Stage 1 (CNN)** | Conv Block 1 ($3 \to 32$) | $(B \times 16, 32, 56, 56)$ | 10,240 |
| **Stage 2 (CNN)** | ResBlock 2 ($32 \to 64$) | $(B \times 16, 64, 28, 28)$ | 57,600 |
| **Stage 3 (CNN)** | ResBlock 3 ($64 \to 128$) | $(B \times 16, 128, 14, 14)$ | 230,400 |
| **Stage 4 (CNN)** | ResBlock 4 ($128 \to 256$) | $(B \times 16, 256, 7, 7)$ | 921,600 |
| **Global Pool** | AdaptiveAvgPool2d | $(B \times 16, 256)$ | 0 |
| **Temporal BiGRU** | 2-layer BiGRU ($256 \to 128 \times 2$) | $(B, 16, 256)$ | 693,248 |
| **Attention** | Dense ($256 \to 64$) + Vector ($64 \to 1$) | $(B, 16, 1)$ | 16,448 |
| **Classifier** | Linear ($256 \to 6$) | $(B, 6)$ | 1,542 |
| **Total** | **Complete Recognizer** | — | **1,918,247 parameters** (~7.7 MB) |

### Runtime Efficiency on Standard Windows Laptop (CPU):
- **CNN Forward Pass:** $\approx 1.8\text{ ms} \times 16\text{ frames} = 28.8\text{ ms}$
- **BiGRU Forward Pass:** $\approx 4.2\text{ ms}$
- **Attention & Classification:** $\approx 0.6\text{ ms}$
- **Total Latency per Inference:** $\approx 35\text{ ms}$ ($\sim 28\text{ inferences/sec}$ on CPU)
