# Interview notes: TactileSense AI

Short answers backed by the project's actual code and results.

**Why OpenCV?**
Tactile images need classical image processing before any learning:
- **Reference subtraction** removes the sensor's static LED illumination, so models see deformation instead of
  lighting.
- **Blur, thresholding and morphology** give a contact mask. Contours and moments then give area, bounding box and
  centroid: interpretable quantities that a robot controller could use directly.
- **HOG** provides an edge-orientation representation for the baseline.
- OpenCV is fast (the whole pipeline plus SVM takes ~5 ms per frame on one CPU thread) and easy to sanity-check.
  The contact area correlates with the robot-measured gel height, r = 0.59.

**Why a classical baseline?**
- It sets the bar a CNN has to clear. Here it turned out to be **as good as the CNN**: test accuracy 0.171 vs
  0.174, with overlapping bootstrap CIs.
- Without it, I might have credited the CNN for something the data simply does not allow.
- It is cheap: 82 s to fit versus ~66 min for the CNN.
- It is diagnosable. Ablations showed that HOG carries the signal and that the model memorises training objects
  (100% train vs 16% validation).

**How does the CNN learn from labelled images?**
- **Input:** the standardised difference image (frame minus reference).
- **Network:** three conv-BN-ReLU blocks extract local edge and curvature patterns, global average pooling makes the
  output tolerant to where the contact is, and a linear layer outputs 10 logits.
- **Training:** cross-entropy between softmax and the digit label, minimised with AdamW over mini-batches of 128.
- **Augmentation:** only physically plausible shifts and pressure scaling.
- **Model selection:** on validation accuracy; early stopping after 5 epochs without improvement (best epoch 11 of 16).

**How is data leakage prevented?**
- **Grouping by object:** all 256 touches of one object share an `object_id`, and splits are made over objects, not
  frames.
- **Official test split:** the test set is the dataset's own split of 100 objects; validation is a grouped,
  stratified 20% of the training objects.
- **Automatic check:** `assert_no_leakage` raises an error if any object appears in two splits.
- **Training-only statistics:** the reference frame is estimated from training frames only.
- **Test set used once:** it is touched only by `src/evaluate.py`, after model selection.
- **Why it matters here:** a random frame-level split inflated the baseline's accuracy from 0.161 to 0.236.

**How to read the confusion matrix and macro-F1**
- Rows are true digits and columns are predictions, normalised per row, so the diagonal is per-class recall.
- Typical confusions involve shared stroke shapes: true "1" predicted "4" (21%), true "9" predicted "4" (30%),
  true "5" predicted "3" (31%).
- Macro-F1 averages per-class F1 with equal weight. Class sizes differ in the test set (494 contact frames for "1"
  versus 919 for "6"), so macro-F1 does not let frequent classes dominate.
- Results: CNN 0.181, SVM 0.159. The difference CI [−0.011, +0.053] includes 0.

**What the model cannot do**
- It does not estimate force, slip, pose or texture; there are no labels for these.
- It does not reliably identify a digit from one touch (~17%). Multiple touches help: 32 touches reach 0.32-0.36.
- It does not generalise to other sensors or gels without a new reference and retraining.
- It does not provide trustworthy confidence: CNN predictions at ≥ 0.9 confidence are right only 18.8% of the time.
- It does not run on hardware: there is no real-time robot integration and no physical sensor testing.

**What I would do with physical tactile sensors**
- Collect data with the labels the robot actually needs: contact pose, normal and shear force (with a force-torque
  sensor), slip events. That would allow regression, not just classification.
- Record touch sequences and the touch position from the robot's kinematics, then fuse them or choose the next touch
  (active perception).
- Capture a fresh no-contact reference per session and gel, and measure robustness across gels and lighting drift.
- Calibrate model confidence (temperature scaling) before using predictions in a controller, and measure on-device
  latency.
