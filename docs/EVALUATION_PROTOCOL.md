# Evaluation Protocol & Leakage Prevention Analysis

**Project:** Indian Sign Language (ISL) Emergency Gesture Recognition  
**Academic Context:** UPES Semester V Deep Learning PBL-DL  
**Objective:** Establish mathematically rigorous, reproducible, leakage-free evaluation protocols.

---

## 1. The Subject Leakage Problem in Video Action Recognition

### The "100% Accuracy" Trap
In early iterations of this project (and commonly in naive deep learning coursework), datasets are split using random clip-level shuffling (e.g. `sklearn.model_selection.train_test_split(test_size=0.20, random_state=42)`).

In the Mendeley ISL emergency dataset:
- Every subject recorded each gesture word across **two distinct takes** (Take 1 and Take 2).
- When videos are shuffled randomly, Take 1 of Signer `005` ends up in the training set, while Take 2 of Signer `005` ends up in the validation/test set.
- **Why this fails scientifically:** Convolutional neural networks have immense capacity. When the same individual appears in both partitions, the network memorizes the signer's unique facial features, clothing, skin tone, jewelry, and background cloth creases rather than learning generalizable spatio-temporal gesture mechanics.
- This produced an **artificial 100.0% validation accuracy**, creating a false illusion of perfect performance that instantly collapsed when presented with a new, unseen human user.

---

## 2. Strict Signer-Independent Group Splitting

To ensure valid generalization, this repository enforces a **Group Disjoint Partitioning Protocol**:

```
                         Complete Dataset (26 Signers, 412 Clips)
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
      Training Partition           Validation Partition          Held-Out Test Partition
     17 Disjoint Signers            4 Disjoint Signers              5 Disjoint Signers
       (002–020 subset)            (001, 004, 009, 021)                (022–026)
          268 clips                      64 clips                       80 clips
               │                            │                            │
               ▼                            ▼                            ▼
     Model Parameter Updates      Epoch Selection & Early Stop     Final One-Shot Scoring
```

### Mathematical Disjointness Guarantee
Let $\mathcal{S}_{\text{train}}$, $\mathcal{S}_{\text{val}}$, and $\mathcal{S}_{\text{test}}$ denote the sets of subject identifiers assigned to each partition. The splitting algorithm asserts:

$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$$
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
$$\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

Furthermore, to prevent duplicate-recording leakage (e.g. when `--include-uncropped` is used where cropped and uncropped copies of the same take exist), each sample tracks an invariant `recording_id = f"{word}_{subject}_{take}"`. The split algorithm strictly verifies that no identical recording ID exists in multiple splits.

---

## 3. The Negative Rejection Class (`none`)

A critical requirement for real-world deployment is rejecting movements that are not emergency signs. If a model only has 5 classes, every wave, cough, or adjustment of glasses will be forced into one of the 5 emergency classes via Softmax.

To teach the network a negative decision boundary:
- Three non-emergency ISL words (`hot`, `lose`, `thief`) from the same dataset are bundled into a 6th class: `none`.
- The test set contains 30 held-out `none` clips from signers 022–026.
- The model achieved **96.67% precision and recall (29/30 correct)** on rejecting these negative gestures, demonstrating that it learns specific emergency movement choreographies rather than triggering on arbitrary hand gestures.

---

## 4. Reproducibility Instructions

To reproduce the held-out test evaluation independently:

```bash
# Activate the local virtual environment
.\.venv\Scripts\activate

# Run evaluation on the 5 unseen test signers (022-026)
python -m src.evaluate --split test

# Run evaluation on the 4 validation tuning signers (001, 004, 009, 021)
python -m src.evaluate --split val
```

Expected Output:
```text
Evaluating on 80 clips from test signers ['022', '023', '024', '025', '026']
              precision    recall  f1-score   support

    accident     1.0000    1.0000    1.0000        10
        call     0.9000    0.9000    0.9000        10
      doctor     1.0000    1.0000    1.0000        10
        help     1.0000    1.0000    1.0000        10
        pain     1.0000    1.0000    1.0000        10
        none     0.9667    0.9667    0.9667        30

    accuracy                         0.9750        80
   macro avg     0.9778    0.9778    0.9778        80
weighted avg     0.9750    0.9750    0.9750        80

Saved eval_results/confusion_matrix.png
Saved eval_results/training_curves.png
Test accuracy on unseen signers: 97.50% (target >= 85%)
```
