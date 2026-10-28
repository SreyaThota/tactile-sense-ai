# Demo checklist

Use this before recording the video or presenting the project live.

## Before the demo

- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Data prepared: `python scripts/prepare_data.py` (creates `data/processed/tactile_mnist_real_64.npz`)
- [ ] Models trained and evaluated: `python scripts/run_experiments.py`.
  These files must exist in `models/`:
  - `preprocess_state.npz`
  - `classical_svm.joblib`
  - `cnn.pt`
  - `training_info.json`
  - `demo_examples.npz`
- [ ] Results package built: `python scripts/build_results.py`
- [ ] Tests green: `python -m pytest -q`
- [ ] Demo starts: `streamlit run app.py`, then open <http://localhost:8501>. The first load takes ~10-20 s while
  PyTorch is imported.

## During the demo

1. **Settings sidebar.** Both models are selected and the input is "Test-set example".
2. **Pick a digit**, e.g. `1`, and press **Another example** until a frame with a clearly visible stroke appears.
3. **Input + contact analysis.** Point out the contact area, the number of blobs and the centroid.
4. **OpenCV preprocessing row.** Walk from left to right through these panels:
   - raw frame
   - no-contact reference
   - grey and HSV views
   - difference map
   - contact overlay
5. **Prediction cards.** Compare the CNN and the HOG + SVM:
   - the predicted digit and its confidence
   - the "correct / wrong" indicator against the true label
   - the class-probability bars
6. **Show a mistake on purpose.** Keep pressing **Another example** until a model is wrong. Explain that one touch
   sees only a fragment of a stroke (see `results/images/prediction_failure_example.png`).
7. **Open the "How the prediction pipeline works" expander.**

## Things to say honestly

- The models classify *which digit was touched*; they do not estimate force, slip or pose.
- Single-frame accuracy is low (see `results/README.md`). Fusing several touches of one object helps.
- No physical sensor was used; all images come from the public Real Tactile MNIST dataset.

## Fallbacks

| Problem | Fix |
|---|---|
| "No trained model found" | run `python scripts/run_experiments.py` |
| "No demo examples found" | run `python -m src.evaluate` (it writes `models/demo_examples.npz`) |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| Uploaded image looks wrong | only GelSight Mini frames from this dataset match the stored reference frame |
