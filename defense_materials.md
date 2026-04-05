# Second Defense Package (April 6, 2026)

## 1. Title Slide

Disease Detection of Mango Leaf Using CNN and Logistic Regression with PSO Optimization

Vidhit Sikri (09814803123) | Preet Kapoor (15414803123) | Samarth (14214803123)

Guide: Dr. Amita Goel | Department of Information Technology, MAIT

## 2. Problem Definition

- Mango leaf diseases reduce crop yield and farmer income.
- Manual inspection is slow, subjective, and difficult in rural regions.
- Need an automated and accurate real-time detection system.

## 3. Gap Identification

| Current Approach               | Limitation                                |
| ------------------------------ | ----------------------------------------- |
| Manual expert inspection       | Subjective, not scalable                  |
| Traditional ML (SVM/RF)        | Handcrafted features miss subtle patterns |
| Deep learning only             | Computationally expensive for mobile      |
| No hyperparameter optimization | Suboptimal performance                    |

Gap: No lightweight hybrid model with automated hyperparameter tuning for mango leaf disease detection.

## 4. Proposed Solution

Architecture:

Input Image -> VGG16 (Feature Extraction) -> Logistic Regression (Classification) -> Disease Output

PSO is used to optimize Logistic Regression hyperparameters (C and max_iter).

Key points:

- Transfer learning from ImageNet via VGG16
- Lightweight classifier (Logistic Regression)
- PSO-based tuning for better performance

## 5. Work Done So Far (Architecture Detail)

1. Input mango leaf image (224 x 224 RGB)
2. Preprocessing
   - Resize to 224 x 224
   - Normalize pixel values
   - Basic augmentation
3. Frozen VGG16 convolutional base
4. Global Average Pooling to produce 512-dimensional feature vector
5. Logistic Regression classifier (PSO-optimized)
6. Final class prediction with confidence score

Classes covered:

- Healthy
- Anthracnose
- Powdery Mildew
- Sooty Mold
- Bacterial Canker
- Rust
- Leaf Blight
- Dieback

## 6. Dataset and Preprocessing

Dataset contains 8 disease classes + healthy leaves.

Preprocessing:

- Resize to 224 x 224
- Normalization
- Gaussian blur (noise reduction)
- RGB to HSV conversion (where applicable)

Augmentation:

- Rotation (+/-20 deg)
- Horizontal/vertical flips
- Brightness and contrast shifts
- Zoom (0.8x to 1.2x)

## 7. Methodology: VGG16 Feature Extraction

Why VGG16:

- Pretrained on 1.2M ImageNet images
- Strong transfer learning behavior
- Stable features with frozen layers

Approach used:

- Keep convolutional layers frozen
- Remove top dense classifier
- Extract feature vectors and use external classifier

Benefit: Very low retraining cost and faster experimentation.

## 8. PSO Hyperparameter Optimization

Search space:

- C in [10^-3, 10^3]
- max_iter in [100, 2000]

Why PSO instead of grid search:

- Faster exploration of large search spaces
- Less brute-force compute
- Handles non-convex behavior well

Reported result:

- With PSO: 98.75% accuracy
- Without PSO: 98.375% accuracy

## 9. Results and Performance

| Metric    | CNN + LR (No PSO) | CNN + LR (With PSO) |
| --------- | ----------------: | ------------------: |
| Accuracy  |           98.375% |              98.75% |
| Precision |             98.2% |               98.6% |
| Recall    |             98.4% |               98.8% |
| F1-Score  |             98.3% |               98.7% |

Comparison with conventional models:

| Model                 | Accuracy |
| --------------------- | -------: |
| SVM                   |    85.2% |
| Random Forest         |    87.3% |
| Proposed hybrid model |   98.75% |

## 10. Expected Outcomes

- Functional AI disease detection system
- High classification accuracy
- Lightweight design suitable for mobile deployment
- Future web/mobile app integration
- Future IoT-assisted monitoring support

## 11. Future Work

Short-term (3 months):

- Mobile app deployment
- Add more disease categories
- Disease severity scoring

Long-term (1 year):

- IoT sensor integration
- Cross-crop adaptation
- Real-time video-based diagnosis

Planned experiments:

- Genetic Algorithm comparison
- Bayesian optimization comparison
- Edge deployment on Raspberry Pi

## 12. Conclusion

- Hybrid VGG16 + Logistic Regression + PSO system built
- Achieved strong benchmark performance
- Outperformed traditional ML baselines
- Practical path to real agricultural deployment

Contributions:

- Vidhit Sikri: Model development, PSO integration, optimization
- Preet Kapoor: Data preprocessing and augmentation pipeline
- Samarth: Testing, comparative evaluation, documentation and PPT

## Individual Contribution Breakdown

| Member       | Roll Number | Contribution                                                    | Evidence Section                    |
| ------------ | ----------- | --------------------------------------------------------------- | ----------------------------------- |
| Vidhit Sikri | 09814803123 | VGG16 implementation, PSO coding, model tuning                  | Methodology and Results             |
| Preet Kapoor | 15414803123 | Data collection, preprocessing, augmentation, LR integration    | Dataset and Preprocessing           |
| Samarth      | 14214803123 | Testing, SVM/RF comparison, documentation, presentation support | Comparative Analysis and References |

## Quick Talking Points

- Problem impact: Manual disease checks are slow and non-scalable.
- System speed: Prediction available in under 2 seconds.
- Core novelty: PSO-guided tuning with VGG16 features + lightweight classifier.
- Performance: 98.75% benchmark-level accuracy.
- Real impact: Early detection can reduce crop losses.

## Likely Viva Questions and Answers

Q: Why PSO over grid search?
A: Grid search is computationally expensive in large spaces. PSO converges with fewer evaluations and gives strong parameter combinations quickly.

Q: Why Logistic Regression after deep features?
A: VGG16 produces highly informative embeddings; Logistic Regression is lightweight, interpretable, and efficient for deployment.

## Final Checklist for April 6, 2026

- [ ] Print 3 hard copies of research paper
- [ ] Keep soft copy in laptop and USB
- [ ] Keep working code in laptop and GitHub backup
- [ ] Keep at least 20 demo images (optional if using synthetic demo)
- [ ] Keep PPT open and ready
- [ ] Formal attire prepared
- [ ] Rehearse individual contribution explanation
