"""
Mango Leaf Disease Detection Demo
Hybrid pipeline: VGG16 feature extraction (real mode) + Logistic Regression + PSO.

This script is defense-ready and can run in two modes:
1) Demo mode (default fallback): Uses deterministic synthetic VGG-like features.
2) Real mode: Uses image folders + VGG16 if OpenCV and TensorFlow are installed.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------- Configuration --------------------
RANDOM_STATE = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DISEASE_CLASSES = [
    "Healthy",
    "Anthracnose",
    "Powdery_Mildew",
    "Sooty_Mold",
    "Bacterial_Canker",
    "Rust",
    "Leaf_Blight",
    "Dieback",
]


@dataclass
class OptimizationResult:
    best_c: float
    best_max_iter: int
    best_val_accuracy: float


# -------------------- Utility --------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)


def safe_optional_imports() -> Dict[str, object]:
    """Import optional heavy dependencies only when available."""
    modules: Dict[str, object] = {}

    try:
        import cv2  # type: ignore

        modules["cv2"] = cv2
    except Exception:
        modules["cv2"] = None

    try:
        from tensorflow.keras.applications import VGG16  # type: ignore
        from tensorflow.keras.applications.vgg16 import preprocess_input  # type: ignore

        modules["VGG16"] = VGG16
        modules["preprocess_input"] = preprocess_input
    except Exception:
        modules["VGG16"] = None
        modules["preprocess_input"] = None

    return modules


# -------------------- Demo Data Generator --------------------
def generate_synthetic_vgg_features(
    samples_per_class: int = 120,
    n_features: int = 512,
    class_count: int = 8,
    label_noise_rate: float = 0.0125,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic, linearly separable synthetic features.

    This mimics VGG feature vectors so the full ML pipeline can be demonstrated
    without requiring any image dataset.
    """
    rng = np.random.default_rng(seed)

    # Class centers are deliberately separated to produce stable demo accuracy.
    centers = rng.normal(loc=0.0, scale=4.5, size=(class_count, n_features))

    features = []
    labels = []

    for class_id in range(class_count):
        cluster = centers[class_id] + rng.normal(
            loc=0.0,
            scale=1.45,
            size=(samples_per_class, n_features),
        )
        features.append(cluster)
        labels.extend([class_id] * samples_per_class)

    x = np.vstack(features)
    y = np.array(labels, dtype=np.int64)

    # Add tiny controlled label noise to avoid unrealistically perfect scores.
    noise_count = int(len(y) * label_noise_rate)
    if noise_count > 0:
        noise_idx = rng.choice(len(y), size=noise_count, replace=False)
        for idx in noise_idx:
            candidates = [c for c in range(class_count) if c != y[idx]]
            y[idx] = int(rng.choice(candidates))

    # Shuffle once for realistic splits.
    indices = rng.permutation(len(y))
    return x[indices], y[indices]


# -------------------- Real Data Loading --------------------
def load_images_from_folder_structure(
    data_dir: str,
    cv2_module: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from:
    data_dir/
      Healthy/
      Anthracnose/
      ...
    """
    cv2 = cv2_module
    images = []
    labels = []

    for class_id, class_name in enumerate(DISEASE_CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            lower = image_name.lower()
            if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.resize(image, IMG_SIZE)
            images.append(image)
            labels.append(class_id)

    if not images:
        raise ValueError("No valid images found in dataset folder structure.")

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)


def extract_vgg16_features(
    images: np.ndarray,
    vgg16_ctor: object,
    preprocess_input_fn: object,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Extract 512-dim features using frozen VGG16 convolutional base + GAP."""
    VGG16 = vgg16_ctor
    preprocess_input = preprocess_input_fn

    model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False

    feature_list = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].astype("float32")
        batch = preprocess_input(batch)
        conv_maps = model.predict(batch, verbose=0)
        pooled = np.mean(conv_maps, axis=(1, 2))
        feature_list.append(pooled)

    return np.vstack(feature_list)


# -------------------- PSO --------------------
def pso_optimize(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    swarmsize: int = 12,
    maxiter: int = 20,
    inertia: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, float]:
    """Minimal PSO implementation to avoid external dependency issues."""
    rng = np.random.default_rng(seed)
    dim = len(lb)

    positions = rng.uniform(lb, ub, size=(swarmsize, dim))
    velocities = rng.normal(0, 0.1, size=(swarmsize, dim))

    pbest_positions = positions.copy()
    pbest_scores = np.array([objective(p) for p in positions])

    best_idx = int(np.argmin(pbest_scores))
    gbest_position = pbest_positions[best_idx].copy()
    gbest_score = float(pbest_scores[best_idx])

    for _ in range(maxiter):
        r1 = rng.random((swarmsize, dim))
        r2 = rng.random((swarmsize, dim))

        velocities = (
            inertia * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )
        positions = np.clip(positions + velocities, lb, ub)

        current_scores = np.array([objective(p) for p in positions])

        improved_mask = current_scores < pbest_scores
        pbest_positions[improved_mask] = positions[improved_mask]
        pbest_scores[improved_mask] = current_scores[improved_mask]

        candidate_idx = int(np.argmin(pbest_scores))
        candidate_score = float(pbest_scores[candidate_idx])
        if candidate_score < gbest_score:
            gbest_score = candidate_score
            gbest_position = pbest_positions[candidate_idx].copy()

    return gbest_position, gbest_score


def optimize_logistic_regression_with_pso(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> OptimizationResult:
    """Tune Logistic Regression hyperparameters C and max_iter via PSO."""

    def objective(params: np.ndarray) -> float:
        log10_c = float(params[0])
        max_iter = int(round(float(params[1])))
        max_iter = int(np.clip(max_iter, 100, 2000))

        c_value = 10 ** log10_c

        try:
            model = LogisticRegression(
                C=c_value,
                max_iter=max_iter,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            accuracy = accuracy_score(y_val, pred)
            return 1.0 - accuracy
        except Exception:
            # Penalize numerically unstable configurations.
            return 1.0

    lower = np.array([-3.0, 100.0])
    upper = np.array([3.0, 2000.0])

    best_params, best_error = pso_optimize(
        objective=objective,
        lb=lower,
        ub=upper,
        swarmsize=10,
        maxiter=15,
        seed=RANDOM_STATE,
    )

    best_c = 10 ** float(best_params[0])
    best_max_iter = int(round(float(best_params[1])))
    best_max_iter = int(np.clip(best_max_iter, 100, 2000))
    best_val_accuracy = 1.0 - float(best_error)

    return OptimizationResult(best_c, best_max_iter, best_val_accuracy)


# -------------------- Training / Evaluation --------------------
def split_and_scale(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=seed,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler


def evaluate_model(model: LogisticRegression, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }

    print("\nEvaluation Metrics")
    print("-" * 50)
    print(f"Accuracy : {metrics['accuracy'] * 100:.4f}%")
    print(f"Precision: {metrics['precision'] * 100:.4f}%")
    print(f"Recall   : {metrics['recall'] * 100:.4f}%")
    print(f"F1-Score : {metrics['f1'] * 100:.4f}%")

    present_labels = sorted(np.unique(y_test).tolist())
    target_names = [DISEASE_CLASSES[i] for i in present_labels]

    print("\nClassification Report")
    print("-" * 50)
    print(classification_report(y_test, pred, labels=present_labels, target_names=target_names, zero_division=0))

    return metrics


# -------------------- Main --------------------
def run_pipeline(data_dir: str, force_demo: bool, samples_per_class: int) -> None:
    print("=" * 70)
    print("MANGO LEAF DISEASE DETECTION")
    print("Pipeline: VGG16 Features + Logistic Regression + PSO")
    print("=" * 70)

    set_seed(RANDOM_STATE)
    optional = safe_optional_imports()

    demo_mode = force_demo
    x_features: Optional[np.ndarray] = None
    y_labels: Optional[np.ndarray] = None

    if not force_demo:
        has_dataset = os.path.isdir(data_dir)
        has_real_stack = optional["cv2"] is not None and optional["VGG16"] is not None

        if has_dataset and has_real_stack:
            print("\nReal mode detected: dataset and required dependencies found.")
            images, labels = load_images_from_folder_structure(data_dir, optional["cv2"])
            print(f"Loaded images: {len(images)}")
            print("Extracting VGG16 features (this can take a few minutes)...")
            x_features = extract_vgg16_features(
                images,
                optional["VGG16"],
                optional["preprocess_input"],
                batch_size=BATCH_SIZE,
            )
            y_labels = labels
        else:
            demo_mode = True

    if demo_mode:
        reason = []
        if force_demo:
            reason.append("forced by --demo")
        if not os.path.isdir(data_dir):
            reason.append(f"dataset folder not found: {data_dir}")
        if optional["cv2"] is None:
            reason.append("opencv not installed")
        if optional["VGG16"] is None:
            reason.append("tensorflow not installed")

        reason_text = ", ".join(reason) if reason else "fallback"
        print("\n[DEMO MODE] Running with synthetic hardcoded feature data")
        print(f"Reason: {reason_text}")

        x_features, y_labels = generate_synthetic_vgg_features(
            samples_per_class=samples_per_class,
            n_features=512,
            class_count=len(DISEASE_CLASSES),
            seed=RANDOM_STATE,
        )
        print(f"Synthetic samples: {len(y_labels)} | Features per sample: {x_features.shape[1]}")

    assert x_features is not None
    assert y_labels is not None

    x_train, x_val, x_test, y_train, y_val, y_test, scaler = split_and_scale(x_features, y_labels)

    print("\nStarting PSO hyperparameter optimization...")
    opt = optimize_logistic_regression_with_pso(x_train, y_train, x_val, y_val)
    print(
        "Best hyperparameters found: "
        f"C={opt.best_c:.6f}, max_iter={opt.best_max_iter}, "
        f"validation_accuracy={opt.best_val_accuracy * 100:.4f}%"
    )

    final_model = LogisticRegression(
        C=opt.best_c,
        max_iter=opt.best_max_iter,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    final_model.fit(x_train, y_train)

    baseline_model = LogisticRegression(
        C=0.2,
        max_iter=100,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    baseline_model.fit(x_train, y_train)

    final_pred = final_model.predict(x_test)
    baseline_pred = baseline_model.predict(x_test)

    final_acc = accuracy_score(y_test, final_pred)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    metrics = evaluate_model(final_model, x_test, y_test)

    print("\nPSO Comparison")
    print("-" * 50)
    print(f"Without PSO: {baseline_acc * 100:.4f}%")
    print(f"With PSO   : {final_acc * 100:.4f}%")
    print(f"Improvement: {(final_acc - baseline_acc) * 100:.4f}%")

    joblib.dump(final_model, "mango_disease_model.pkl")
    joblib.dump(scaler, "feature_scaler.pkl")
    joblib.dump(
        {
            "classes": DISEASE_CLASSES,
            "demo_mode": demo_mode,
            "best_c": float(opt.best_c),
            "best_max_iter": int(opt.best_max_iter),
            "validation_accuracy": float(opt.best_val_accuracy),
            "test_metrics": {k: float(v) for k, v in metrics.items()},
        },
        "run_metadata.pkl",
    )

    print("\nSaved artifacts")
    print("-" * 50)
    print("mango_disease_model.pkl")
    print("feature_scaler.pkl")
    print("run_metadata.pkl")

    print("\nExecution note")
    print("-" * 50)
    print("This run verifies end-to-end pipeline execution, including synthetic evaluation when image data is unavailable.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mango disease detection demo pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="mango_leaf_dataset",
        help="Path to real image dataset root folder",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force synthetic demo mode even if dataset exists",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=120,
        help="Synthetic samples per class in demo mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        force_demo=args.demo,
        samples_per_class=max(30, args.samples_per_class),
    )
