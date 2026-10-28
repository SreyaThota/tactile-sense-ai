# TactileSense AI: demo video script (about 110 seconds)

All visuals are real outputs from this repository. Start the app with `streamlit run app.py` before recording;
see [demo_checklist.md](demo_checklist.md).

| Time | On screen | Asset |
|---|---|---|
| 0-15 s | Title slide, then the raw tactile frame | `results/images/original_tactile_image.png` |
| 15-35 s | OpenCV pipeline figure, then the contact mask | `results/images/opencv_preprocessing.png`, `results/images/contact_mask.png` |
| 35-55 s | Architecture diagram (Mermaid block in the README), then the training curves | `README.md`, `results/plots/training_curves.png` |
| 55-80 s | Live Streamlit app: pick digit, press **Another example**, show both prediction cards | app at `http://localhost:8501` (stills: `results/images/streamlit_demo*.png`) |
| 80-100 s | Model comparison, then the confusion matrices, then the fusion curve | `results/plots/model_comparison.png`, `confusion_matrix_*.png`, `multi_touch_fusion.png` |
| 100-120 s | Failure case figure, then the GitHub repo page | `results/images/prediction_failure_example.png` |

---

### 0-15 s: the problem
**Show:** a raw GelSight Mini frame.

**Say:**
> "This is an image from a GelSight Mini, a vision-based tactile sensor: a camera behind a soft, illuminated gel.
> When the gel presses on an object, the shading changes. TactileSense AI asks a simple question: from a single
> touch like this, can we tell which object the robot is touching?"

### 15-35 s: OpenCV preprocessing
**Show:** `opencv_preprocessing.png`, panels left to right, then `contact_mask.png`.

**Say:**
> "The raw frame is dominated by the sensor's static LED lighting. So I estimate a no-contact reference, the median
> of training frames, and subtract it. What remains is the deformation. A Gaussian blur, a fixed threshold and a
> morphological opening give the contact mask. From that I get contours, a bounding box and the centroid. As a check,
> the detected contact area correlates with the gel height the robot measured: r of 0.59."

### 35-55 s: the two models
**Show:** the architecture diagram, then `training_curves.png`.

**Say:**
> "I compare two supervised models. The baseline uses HOG and contact features with an SVM. The second is a compact
> CNN, about 290 thousand parameters, trained on the difference images. Augmentation is limited to shifts and signal
> scaling, with no flips, because the lighting direction is fixed. The data is split by physical object, so the 100
> test digits are never seen in training. The checkpoint is chosen on validation accuracy: here, epoch 11."

### 55-80 s: live prediction
**Show:** the Streamlit app.
1. Choose a digit.
2. Press **Another example** a few times.
3. Point at the contact metrics, the preprocessing row and the two prediction cards.

**Say:**
> "In the demo I pick a held-out test frame. The app shows the contact area and centroid, the full preprocessing
> chain, and both models' class probabilities, with the true label for comparison. As you'll see, the models are
> often wrong on a single touch."

(Show both a correct and a wrong case; do not stage only correct ones.)

### 80-100 s: results
**Show:** `model_comparison.png`, then the two confusion matrices, then `multi_touch_fusion.png`.

**Say:**
> "On the held-out objects, both models reach about 17% single-frame accuracy against 10% chance. Bootstrap
> intervals over the test objects show both are above chance, but neither is reliably better than the other. The
> reason is visible in the data: one touch usually sees only a fragment of a stroke. If I average the predictions of
> 32 touches of the same object, accuracy roughly doubles, to 32 to 36 percent."

### 100-120 s: contribution and limitations
**Show:** `prediction_failure_example.png`, then the repository.

**Say:**
> "Here the CNN called a 6 an 8 with 99.7% confidence. The robot's gel height shows the sensor actually pressed the
> mat, not the digit. So the contact detector and the CNN's calibration both need work. The baseline analysis also
> showed that a frame-level split would have inflated accuracy by almost half. The project is a complete,
> reproducible tactile learning pipeline with honest evaluation. The next steps are multi-touch fusion, confidence
> calibration, and testing on a physical sensor."
