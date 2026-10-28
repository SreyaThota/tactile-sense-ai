# TactileSense AI

Vision-based tactile perception with supervised learning.

TactileSense AI classifies images from a vision-based tactile sensor (GelSight Mini) using an
OpenCV preprocessing pipeline, a classical machine learning baseline (HOG features + SVM), and a
compact PyTorch CNN, evaluated on a held-out test set.

## Status

Work in progress. Setup, training, evaluation and demo instructions, as well as results, will be
added once the pipeline has been implemented and run on real data.

## Planned components

- OpenCV preprocessing and contact-region visualization
- Classical baseline: HOG / intensity features + scikit-learn SVM
- CNN classifier in PyTorch (CPU-friendly)
- Leakage-free train / validation / test splits grouped by physical object
- Evaluation report: accuracy, macro-F1, confusion matrix, latency
- Streamlit demo

## Dataset

Planned: [Real Tactile MNIST](https://huggingface.co/TimSchneider42) (Schneider et al., 2025,
arXiv:2506.06361), real GelSight Mini touches of 3D-printed MNIST digits, CC-BY-2.0.
Data is downloaded by a script and is not committed to this repository.
