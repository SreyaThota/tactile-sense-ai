"""Train tiny models on SYNTHETIC data, save them, reload them and run inference.

This checks the training -> checkpoint -> inference plumbing only; the synthetic task is trivial
and its accuracy is not a tactile-sensing result.
"""

import copy

import numpy as np
import pytest

from src.config import load_config
from src.inference import ModelNotAvailableError, TactilePredictor
from src.models import TactileCNN, count_parameters
from src.train import run_training


@pytest.fixture
def trained_models(tmp_path, folder_dataset):
    cfg = copy.deepcopy(load_config())
    cfg["data"].update(source="folder", folder_root=str(folder_dataset), val_fraction=0.25, test_fraction=0.34)
    cfg["classical"].update(max_train_samples=1000, pca_components=8)
    cfg["cnn"].update(max_epochs=2, batch_size=8, num_threads=1)
    cfg["paths"].update(models_dir=str(tmp_path / "models"), reports_dir=str(tmp_path / "reports"))
    info = run_training(cfg)
    return tmp_path / "models", info


def test_cnn_output_shape():
    import torch

    model = TactileCNN(num_classes=7)
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 7)
    assert count_parameters(model) > 0


def test_training_writes_checkpoints(trained_models):
    models_dir, info = trained_models
    for name in ("preprocess_state.npz", "classical_svm.joblib", "cnn.pt", "training_info.json"):
        assert (models_dir / name).is_file()
    assert info["cnn"]["epochs_run"] >= 1
    assert info["data"]["splits"]["test"]["frames_used"] > 0


def test_predictor_loads_and_predicts(trained_models, folder_dataset):
    from src.preprocessing import load_image

    models_dir, _ = trained_models
    predictor = TactilePredictor(models_dir)
    assert set(predictor.available_models) == {"cnn", "classical"}
    image = load_image(next((folder_dataset / "vertical").glob("*.png")))
    for model in predictor.available_models:
        pred = predictor.predict(image, model)
        assert pred.label in predictor.class_names
        assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
        assert 0 <= pred.confidence <= 1
        assert pred.contact.has_contact


def test_predict_proba_batch(trained_models, folder_dataset):
    from src.data_loader import load_image_folder

    predictor = TactilePredictor(trained_models[0])
    ds = load_image_folder(folder_dataset, 64)
    probs = predictor.predict_proba(ds.images[:5], "cnn")
    assert probs.shape == (5, 2)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)
    with pytest.raises(ValueError):
        predictor.predict_proba(ds.images[:1], "unknown")


def test_missing_models_give_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        TactilePredictor(tmp_path)
