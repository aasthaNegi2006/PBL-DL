# Technical Viva Voce Defense Guide (25 Core Questions & Answers)

**Course:** Deep Learning (Semester V) — Project-Based Evaluation (PBE)  
**Date:** Monday, 23 November 2026  
**Candidate:** Aastha Negi  
**Project:** Indian Sign Language (ISL) Emergency Gesture Recognition System  

---

### Q1: Why did your original model report 100% accuracy, and why was that flawed?
**Answer:** The original model split videos using random clip-level shuffling. In the dataset, each signer performed two takes of every sign. Random splitting placed Take 1 in the training set and Take 2 in the test set for the same individual. The deep network memorized signer-specific attributes (clothing, skin tone, face, and studio background cloth creases) rather than generalizable gesture mechanics, causing identity leakage. When evaluated on 5 completely unseen signers (`022`–`026`), true generalization accuracy is **97.50%** (0.9778 Macro F1), which is genuine and honest.

---

### Q2: Why use a 2D-CNN + BiGRU + Attention architecture instead of a 3D-CNN or pure Transformer?
**Answer:**
1. **Computational Efficiency:** 3D-CNNs (e.g. C3D, I3D) involve spatio-temporal 3D convolutions with high parameter counts and FLOPs, making real-time CPU inference on standard laptops impossible ($>200\text{ ms/clip}$).
2. **Temporal Dynamics:** 2D-CNN extracts frame-level spatial features ($256$-dim), while a 2-layer Bidirectional GRU captures forward and backward motion trajectories with only 1.9M parameters.
3. **Temporal Saliency:** Bahdanau Attention dynamically identifies key contact frames (e.g., fists colliding in `accident`) rather than uniform pooling.
4. **Data Scale:** Vision Transformers (ViViT/TimeSformer) require hundreds of thousands of video samples to avoid severe overfitting; our hybrid recurrent model regularizes effectively on hundreds of clips.

---

### Q3: Explain the mathematical mechanism of Bahdanau Additive Attention in your model.
**Answer:**
Given the BiGRU hidden sequence $H = [h_1, h_2, \dots, h_{16}] \in \mathbb{R}^{16 \times 256}$:
1. An alignment energy scalar is computed for each frame:
   $$e_t = v_a^\top \tanh(W_a h_t + b_a), \quad W_a \in \mathbb{R}^{64 \times 256}, \, v_a \in \mathbb{R}^{64}$$
2. The energies are normalized into an attention distribution via Softmax:
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{16} \exp(e_k)}, \quad \text{where } \sum_{t=1}^{16} \alpha_t = 1$$
3. The final context vector is the convex combination:
   $$c = \sum_{t=1}^{16} \alpha_t h_t \in \mathbb{R}^{256}$$
This allows the classification head to focus on key motion peaks rather than being dominated by the final frame.

---

### Q4: Why did you choose GRU over LSTM?
**Answer:** GRU has two gates (Reset $r_t$ and Update $u_t$) compared to LSTM's three gates (Input, Forget, Output) plus a separate cell state. GRU requires ~25% fewer parameters, trains faster, is less prone to overfitting on medium-sized datasets, and achieves comparable sequence modeling capability for short 16-frame gesture sequences.

---

### Q5: What loss function did you use and how is backpropagation computed?
**Answer:** Multi-class Cross-Entropy Loss with L2 regularization (weight decay $\lambda = 10^{-4}$):
$$\mathcal{L} = -\sum_{k=1}^K y_k \log \hat{p}_k + \lambda \|\theta\|_2^2$$
Where $\hat{p}_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$. The gradient with respect to logit $z_i$ simplifies to:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{p}_i - y_i$$
Backpropagation propagates this error vector back through the linear classifier, the attention weighting Jacobian, the recurrent unrolled BiGRU time-steps via Backpropagation Through Time (BPTT), and finally through the CNN convolutional layers.

---

### Q6: What is the purpose of the 6th `none` class?
**Answer:** Without a negative rejection class, the Softmax activation forces every input to sum to 1.0, causing random hand movements (scratching nose, waving, resting) to be misclassified as one of the 5 emergency signs. We trained a 6th `none` class using negative gestures (`hot`, `lose`, `thief`), achieving 96.67% precision in rejecting non-emergency movements.

---

### Q7: Why did the live webcam initially trigger "Help" or "Accident" on idle frames?
**Answer:**
1. **Domain Gap:** Dataset training clips were recorded against a dark blue backdrop with isolated hand motions. In a room webcam, resting hands or ambient camera noise produced small non-zero gradients.
2. **Lack of Static Backgrounds in Training:** Because the dataset lacked static rest videos, unconstrained ConvNet + BiGRU outputs defaulted to Logit 0 (`accident` $= +3.94$).
3. **The Solution:** We implemented:
   - Calibrated motion energy thresholding ($\theta_{\text{motion}} = 2.4$), returning `idle` without calling the network when hands are at rest.
   - Ambiguity margin filtering ($\text{top}_1 - \text{top}_2 \ge 0.12$).
   - Torso/hands ROI framing ($c_x=0.50, c_y=0.58, \text{size}=0.75$) excluding face glare.

---

### Q8: How does the confirmation state machine prevent false positive triggers?
**Answer:** A single frame above the confidence threshold does not immediately trigger speech. The system requires $N=3$ consecutive predictions of the *exact same candidate class* ($pred\_name == candidate\_class$) with confidence $\ge 0.75$ and margin over runner-up $\ge 0.12$. If the class changes or motion drops below threshold, the counter resets. Additionally, a 3-second cooldown prevents repeated voice announcements.

---

### Q9: What learning rate schedule was used and why?
**Answer:** We used **Cosine Annealing Learning Rate Decay** (`torch.optim.lr_scheduler.CosineAnnealingLR`) starting at $\eta_{\max} = 3 \times 10^{-4}$ down to $\eta_{\min} = 10^{-6}$. Unlike step decay, cosine annealing provides smooth, continuous parameter updates that allow aggressive exploration in early epochs followed by fine convergence into a flat local minimum.

---

### Q10: How do you prevent overfitting?
**Answer:**
1. **Signer-Independent Group Splitting:** Completely disjoint signers in train, val, and test.
2. **Spatial Data Augmentation:** Random brightness ($\pm 15\%$), contrast ($\pm 15\%$), and spatial translations ($\pm 6\text{ px}$).
3. **Temporal Jittering:** Training clips are randomly sampled across temporal offsets rather than strict uniform intervals.
4. **Weight Decay:** L2 regularization $\lambda = 10^{-4}$ penalizes large weights.
5. **Early Stopping:** Best model selected on validation loss, not training accuracy.

---

### Q11: Explain the confusion matrix on the held-out test signers.
**Answer:** Across 80 clips (10 `accident`, 10 `call`, 10 `doctor`, 10 `help`, 10 `pain`, 30 `none`):
- `accident`, `doctor`, `help`, `pain` achieved **10/10 correct (100% recall)**.
- `call` achieved **9/10 correct (90% recall)**, with 1 clip confused as `none` due to subtle finger phone-curve motion by Signer 024.
- `none` achieved **29/30 correct (96.7% recall)**, with 1 clip misclassified as `call`.
- Overall accuracy: **97.50%**, Macro F1: **0.9778**.

---

### Q12: How are video frames preprocessed for the model?
**Answer:**
1. Center/torso square cropping to match dataset framing.
2. Bilinear resize to $112 \times 112$ pixels.
3. Conversion from BGR (OpenCV) to RGB.
4. Normalization: scale to $[0, 1]$, then apply ImageNet standardization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
5. Output shape: $(16, 3, 112, 112)$.

---

### Q13: Why did you pick 16 frames per sequence?
**Answer:** The median duration of an ISL emergency gesture in the dataset is 2.56 seconds ($\approx 75$ frames at 30 FPS). Squeezing a complete sign into 16 evenly spaced frames provides a temporal resolution of $\approx 6.25\text{ FPS}$, which captures macro trajectory (hand rise, formation, collision/tap, retraction) while maintaining sub-40 ms inference latency on CPU.

---

### Q14: What is the purpose of the Residual connection in your ConvNet?
**Answer:** In standard deep convolutional networks, gradients degrade during backpropagation through repeated matrix multiplications (vanishing gradient problem). Residual connections introduce an identity shortcut:
$$y = \mathcal{F}(x) + W_s x$$
During backpropagation: $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + W_s \right)$. The additive identity term ensures gradient flow directly back to early layers without attenuation.

---

### Q15: How does your system ensure security and safe checkpoint loading?
**Answer:** In PyTorch, calling `torch.load()` without constraints can lead to arbitrary code execution if weights contain malicious pickle opcodes. We strictly enforce `weights_only=True` across all loading paths in `src/inference.py` and `src/dataset.py`, and verify SHA-256 digests for model provenance.

---

### Q16: How does the rolling frame window work in real-time inference?
**Answer:** The `FrameWindow` data structure maintains a timestamped FIFO buffer of frames covering the last $2.5\text{ seconds}$ (the median duration of a full sign). Rather than sending the last 16 consecutive frames (which would only represent $\approx 0.5\text{s}$ of movement), the system samples 16 evenly distributed frames across the entire 2.5-second buffer.

---

### Q17: What is the difference between Macro F1 and Weighted F1?
**Answer:**
- **Macro F1:** Computes the arithmetic mean of F1 scores across all classes without weighting by sample support: $\text{Macro F1} = \frac{1}{K}\sum_{k=1}^K F1_k$. It treats all classes equally, penalizing poor performance on minority classes.
- **Weighted F1:** Weights each class F1 by its support $N_k / N$. Because our held-out test set has 30 `none` clips and 10 of each emergency class, Macro F1 (0.9778) provides a more rigorous indicator of unweighted emergency recognition capability.

---

### Q18: What is the time complexity of the forward pass?
**Answer:**
- ConvNet: $\mathcal{O}(T \cdot \sum_l K_l^2 \cdot C_{l-1} \cdot C_l \cdot H_l \cdot W_l) \approx 1.2 \times 10^8\text{ FLOPs}$
- BiGRU: $\mathcal{O}(T \cdot 2 \cdot (3 \cdot (C_{\text{in}} \cdot H + H^2))) \approx 3.9 \times 10^5\text{ FLOPs}$
- Attention: $\mathcal{O}(T \cdot (H \cdot D_a + D_a)) \approx 3.3 \times 10^4\text{ FLOPs}$
- Classification: $\mathcal{O}(2H \cdot K) \approx 3 \times 10^3\text{ FLOPs}$
Total operations per inference: $\sim 1.2 \times 10^8\text{ FLOPs}$, completing in under $35\text{ ms}$ on an Intel Core i5/i7 CPU.

---

### Q19: How did you select the best model checkpoint?
**Answer:** Checkpoints were monitored at every training epoch on the disjoint validation partition (Signers `001`, `004`, `009`, `021`). Epoch 34 was selected because it achieved the minimum validation Cross-Entropy loss ($0.2522$) with $100\%$ validation accuracy, avoiding late-stage overfitting.

---

### Q20: What are the primary ethical considerations of this system?
**Answer:**
1. **Safety Warning:** Clear disclosure that the system is an academic research prototype, not a certified medical or life-safety dispatcher.
2. **Signer Inclusivity:** Tested across diverse male and female signers of varying skin tones and signing speeds to prevent algorithmic bias.
3. **Accessibility:** Designed with both visual feedback (HUD & web progress bars) and acoustic feedback (TTS) to facilitate DHH-hearing cross-communication.

---

### Q21: How does AdamW optimizer differ from standard Adam?
**Answer:** Standard Adam applies L2 weight regularization to the loss function, which causes the weight decay term to be adapted by the running historical gradient variances ($v_t$). AdamW decouples weight decay from the gradient update:
$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
This prevents weights with large historical gradients from escaping regularization.

---

### Q22: What happens if an unseen signer performs a gesture with their left hand instead of right?
**Answer:** In the training dataset, gestures were primarily recorded with right-hand dominance. For mirror-image or left-handed signers, spatial features reflect horizontally. During training, random small translations were used, but full horizontal flipping was avoided because asymmetric signs (e.g. `call` to the right ear vs `doctor` tapping left wrist) carry lateral semantics. Supporting bilateral signing would require left-hand training examples or landmark mirroring.

---

### Q23: Why is Global Average Pooling preferred over Flattening before the BiGRU?
**Answer:** If feature maps of shape $7 \times 7 \times 256$ were flattened directly, the linear layer would require $(7 \times 7 \times 256) \times 256 \approx 3.2\text{ million parameters}$, drastically increasing risk of overfitting and memory usage. Global Average Pooling computes the spatial mean across each channel:
$$z_{t,c} = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W A_{t,c,i,j}$$
This reduces the spatial embedding to exactly 256 values with zero trainable parameters and provides translation invariance.

---

### Q24: What are the main failure modes of the system?
**Answer:**
1. **Incomplete Signs:** When a user aborts halfway through a sign, the temporal sequence lacks the resolution phase.
2. **Extreme Lighting Variations:** Underexposure (< 10 lux) degrades hand contour contrast.
3. **Severe Perspective Distortion:** Signing too close to the camera (< 0.5m) clips hands outside the ROI bounding box.

---

### Q25: How would you extend this project for a Master's thesis or production system?
**Answer:**
1. **Multimodal Fusion:** Combine RGB video frames with MediaPipe/OpenPose 3D hand landmarks via cross-attention.
2. **Continuous Sign Language Recognition (CSLR):** Replace isolated classification with Connectionist Temporal Classification (CTC) loss or a sequence-to-sequence Transformer with beam search.
3. **Edge Deployment:** Quantize the PyTorch model to INT8 using ONNX Runtime or TensorRT for low-power edge cameras (Raspberry Pi 5 / Jetson Nano).
