# Resume bullets: TactileSense AI

- Built a Python/OpenCV pipeline for real GelSight Mini tactile images (153,600 frames). It uses reference
  subtraction, contour-based contact segmentation and HOG features, and the contact detector was validated against
  robot-measured gel height (r = 0.59).
- Trained and compared a calibrated HOG + SVM baseline and a 289k-parameter PyTorch CNN on object-grouped splits.
  Both reached ~17% single-touch accuracy on 100 unseen objects (chance 10%), and fusing 32 touches raised this to
  32-36%. A leakage control showed that a frame-level split would have inflated accuracy by 46%.
- Delivered a reproducible, tested ML project: YAML-configured pipeline, 26 pytest tests, bootstrap confidence
  intervals, failure and calibration analysis, and a Streamlit demo for interactive tactile image inference.
