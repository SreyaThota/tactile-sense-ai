"""TactileSense AI - interactive demo.

Run from the repository root:  streamlit run app.py
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config, resolve
from src.inference import ModelNotAvailableError, TactilePredictor
from src.preprocessing import contact_area_batch, decode_image, image_statistics, to_rgb
from src.visualization import COLORS, LABELS, plot_preprocessing_steps

st.set_page_config(page_title="TactileSense AI", page_icon=":material/touch_app:", layout="wide")

CFG = load_config()


@st.cache_resource(max_entries=2)
def get_predictor(models_dir: str) -> TactilePredictor:
    return TactilePredictor(models_dir)


@st.cache_data(max_entries=1, show_spinner="Loading held-out test frames...")
def load_test_examples(cache_path: str, models_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Test-split frames with detected contact, read from the local dataset cache (if present)."""
    path = resolve(cache_path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        test = z["official_split"] == "test"
        images, labels = z["images"][test], z["labels"][test]
    predictor = get_predictor(models_dir)
    area = contact_area_batch(images, predictor.reference, predictor.pre_cfg)
    keep = area >= predictor.pre_cfg["min_contact_fraction"]
    return images[keep], labels[keep]


def enlarge(image: np.ndarray, size: int = 256) -> np.ndarray:
    # Nearest-neighbour keeps the 64x64 pixels crisp instead of blurring them for display.
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)


st.title("TactileSense AI")
st.caption(
    "Digit classification from single GelSight Mini tactile images "
    "(Real Tactile MNIST, Schneider et al. 2025, CC-BY-2.0). "
    "The models predict *which 3D-printed digit was touched*; they do not estimate contact force."
)

models_dir = str(resolve(CFG["paths"]["models_dir"]))
try:
    predictor = get_predictor(models_dir)
except ModelNotAvailableError as err:
    st.error(str(err), title="No trained model found", icon=":material/error:")
    st.code("python scripts/prepare_data.py\npython scripts/run_experiments.py", language="bash")
    st.stop()

if not predictor.available_models:
    st.error("The preprocessing state exists but no classifier has been trained yet.", icon=":material/error:")
    st.stop()

with st.sidebar:
    st.header("Settings")
    models = st.pills(
        "Models", predictor.available_models, selection_mode="multi",
        default=predictor.available_models, format_func=lambda m: LABELS[m],
    )
    source = st.segmented_control("Input", ["Test-set example", "Upload image"], default="Test-set example")

image, true_label = None, None
if source == "Upload image":
    upload = st.file_uploader("Tactile sensor image", type=["png", "jpg", "jpeg", "bmp"])
    if upload is not None:
        try:
            image = decode_image(upload.getvalue())
        except ValueError:
            st.error("The file could not be decoded as an image.", icon=":material/error:")
    st.caption(
        f"Images are resized to {predictor.image_size}x{predictor.image_size}. The no-contact reference "
        "frame was estimated from GelSight Mini training data, so images from other sensors or "
        "lighting set-ups will not be processed meaningfully."
    )
else:
    examples = load_test_examples(CFG["data"]["cache_path"], models_dir)
    if examples is None:
        st.warning("Dataset cache not found. Run `python scripts/prepare_data.py` or upload an image.",
                   icon=":material/warning:")
    else:
        ex_images, ex_labels = examples
        with st.container(horizontal=True, vertical_alignment="bottom"):
            digit = st.selectbox("True digit", predictor.class_names)
            if "example_seed" not in st.session_state:
                st.session_state.example_seed = 0
            if st.button("Another example", icon=":material/shuffle:"):
                st.session_state.example_seed += 1
        candidates = np.flatnonzero(ex_labels == predictor.class_names.index(digit))
        pick = candidates[np.random.default_rng(st.session_state.example_seed).integers(len(candidates))]
        image, true_label = ex_images[pick], digit
        st.caption("Frame from the official test split (objects never used for training or model selection).")

if image is None:
    st.info("Choose a test-set example or upload a tactile image to start.", icon=":material/info:")
    st.stop()

prepared = predictor.prepare(image)
contact = predictor.contact(prepared)

left, right = st.columns([1, 2])
with left:
    st.subheader("Input")
    st.image(to_rgb(enlarge(prepared)), caption="Tactile frame (enlarged)", alt="Input tactile image")
with right:
    st.subheader("Contact analysis")
    with st.container(horizontal=True):
        st.metric("Contact area", f"{contact.area_fraction:.1%}", border=True)
        st.metric("Contact blobs", contact.num_regions, border=True)
        st.metric("Centroid (x, y)",
                  "-" if contact.centroid is None else f"{contact.centroid[0]:.0f}, {contact.centroid[1]:.0f}",
                  border=True)
    if not contact.has_contact:
        st.warning("No contact detected: the gel looks undeformed, so the image carries no information "
                   "about the touched object and any prediction below is essentially a guess.",
                   title="No contact", icon=":material/warning:")
    stats = image_statistics(prepared)
    st.caption(f"Grey mean {stats['gray_mean']:.1f}, std {stats['gray_std']:.1f}, "
               f"range {stats['gray_min']:.0f}-{stats['gray_max']:.0f}")

st.subheader("OpenCV preprocessing")
fig = plot_preprocessing_steps(prepared, predictor.reference, contact, predictor.pre_cfg["blur_kernel"])
st.pyplot(fig, alt="Preprocessing steps: raw, reference, grey, hue, difference, contact overlay")

st.subheader("Prediction")
if not models:
    st.info("Select at least one model in the sidebar.", icon=":material/info:")
for column, model in zip(st.columns(max(len(models), 1)), models):
    pred = predictor.predict(prepared, model)
    with column, st.container(border=True):
        delta = None
        if true_label is not None:
            delta = "correct" if pred.label == true_label else f"wrong (true {true_label})"
        st.metric(LABELS[model], f"digit {pred.label}", delta=delta,
                  delta_color="normal" if pred.label == true_label else "inverse")
        st.caption(f"Confidence {pred.confidence:.1%}")
        probs = pd.DataFrame({"digit": list(pred.probabilities), "probability": list(pred.probabilities.values())})
        st.bar_chart(probs, x="digit", y="probability", color=COLORS[model], height=220,
                     alt=f"{LABELS[model]} class probabilities")

with st.expander("How the prediction pipeline works"):
    st.markdown(
        """
1. **Resize** the frame to the training resolution (64x64, `INTER_AREA`).
2. **Reference subtraction**: a no-contact reference (pixel-wise median of training frames) is
   subtracted. This removes the sensor's static LED illumination so only gel deformation remains.
3. **Contact segmentation**: Gaussian blur, fixed threshold on the max channel difference,
   morphological opening, then contours, bounding box and centroid.
4. **Classification**
   - *HOG + SVM*: HOG of the grey difference image, deformation histogram, colour statistics and
     contact geometry -> standardisation -> PCA -> RBF-SVM with sigmoid-calibrated probabilities.
   - *CNN*: the standardised 3-channel difference image goes through a compact 3-block CNN.
5. A single touch often covers only part of a digit, so single-frame predictions are uncertain;
   averaging over several touches of the same object (see `reports/`) is much more reliable.
"""
    )
