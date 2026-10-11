# =================================================================================================
# PSL MAJOR REVISION — MASTER STRATIFIED BASELINE TRAINER
# Single Jupyter Cell
#
# CLEAN BASELINE:
#   NO data augmentation
#   NO MixUp
#   NO CutMix
#   NO TTA
#   NO artificial perturbations
#
# Automatically:
#   - trains all decided seeds
#   - checkpoints every epoch
#   - resumes interrupted seed from latest epoch
#   - skips completed seeds
#   - evaluates best validation checkpoint on test set
#   - records paper/reviewer metrics and computational cost
#   - aggregates mean ± SD and 95% CI
#
# Dataset:
#   /home/kamran/Datasets_for_Major_Revision/Combined-datasets-splitts
# =================================================================================================


# =================================================================================================
# 0. USER SETTING — CHANGE ONLY THIS FOR THE NEXT MODEL
# =================================================================================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    choices=[
        "deit_base",
        "convnext_base",
        "swin_base",
        "efficientnetv2_b3",
        "mobilenetv4_medium",
    ],
)
args = parser.parse_args()

MODEL = args.model


#MODEL = "deit_base"

# Valid values:
# "deit_base"
# "convnext_base"
# "swin_base"
# "efficientnetv2_b3"
# "mobilenetv4_medium"


# =================================================================================================
# 1. IMPORTS
# =================================================================================================

import os
import sys
import csv
import json
import math
import time
import random
import socket
import platform
import traceback
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile

import timm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Optional FLOPs support
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except Exception:
    FVCORE_AVAILABLE = False

# Optional Student-t CI support
try:
    from scipy.stats import t as student_t
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


ImageFile.LOAD_TRUNCATED_IMAGES = True


# =================================================================================================
# 2. FIXED EXPERIMENT CONFIGURATION
# =================================================================================================
DATA_ROOT = Path(
    "/home/kamran/Datasets_for_Major_Revision/"
    "LODO-3_CFRT_Images2021_to_UAlpha40"
)


RESULTS_ROOT = Path(
    "/home/kamran/Datasets_for_Major_Revision/"
    "revision_results/LODO-3_CFRT_Images2021_to_UAlpha40"
)


"""
DATA_ROOT = Path(
    "/home/kamran/Datasets_for_Major_Revision/Combined-datasets-splitts"
)

RESULTS_ROOT = Path(
    "/home/kamran/Datasets_for_Major_Revision/revision_results/stratified_baseline"
)
"""
EXPECTED_CLASSES = 39

SEEDS = [
    42,
    123,
    2026,
    7777,
    ]

EPOCHS = 50
WARMUP_EPOCHS = 5

BATCH_SIZE = 80
NUM_WORKERS = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05

IMAGE_SIZE = 224
RESIZE_SIZE = 256

# Fixed training budget:
# No early stopping.
#
# Best validation checkpoint is still selected.

USE_AMP = False

# For Tesla P40, FP32 provides a clean and consistent baseline.
# AMP can be evaluated separately later if deployment optimization is desired.

SAVE_CHECKPOINT_EVERY_EPOCH = True

INFERENCE_WARMUP = 30
INFERENCE_ITERATIONS = 100


# =================================================================================================
# 3. MODEL REGISTRY
# =================================================================================================

MODEL_REGISTRY = {

    "deit_base": {
        "paper_name": "DeiT-Base",
        "timm_name": "deit_base_patch16_224.fb_in1k",
    },

    "convnext_base": {
        "paper_name": "ConvNeXt-Base",
        "timm_name": "convnext_base.fb_in22k_ft_in1k",
    },

    "swin_base": {
        "paper_name": "Swin-Base",
        "timm_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    },

    "efficientnetv2_b3": {
        "paper_name": "EfficientNetV2-B3",
        "timm_name": "tf_efficientnetv2_b3.in21k_ft_in1k",
    },

    "mobilenetv4_medium": {
        "paper_name": "MobileNetV4-Conv-Medium",
        "timm_name": "mobilenetv4_conv_medium.e500_r224_in1k",
    },
}


if MODEL not in MODEL_REGISTRY:
    raise ValueError(
        f"Invalid MODEL '{MODEL}'. "
        f"Choose from: {list(MODEL_REGISTRY.keys())}"
    )

MODEL_INFO = MODEL_REGISTRY[MODEL]

PAPER_MODEL_NAME = MODEL_INFO["paper_name"]
TIMM_MODEL_NAME = MODEL_INFO["timm_name"]


# =================================================================================================
# 4. PATHS
# =================================================================================================

MODEL_RESULTS_DIR = RESULTS_ROOT / MODEL
MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
TEST_DIR  = DATA_ROOT / "test"


# =================================================================================================
# 5. BASIC VALIDATION BEFORE TRAINING
# =================================================================================================

for required_path in [DATA_ROOT, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{required_path}"
        )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not detected. "
        "This experiment is intended for GPU execution."
    )

DEVICE = torch.device("cuda")


# =================================================================================================
# 6. REPRODUCIBILITY
# =================================================================================================

def set_global_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =================================================================================================
# 7. JSON UTILITIES
# =================================================================================================

def json_safe(obj):

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    if isinstance(obj, tuple):
        return list(obj)

    return obj


def save_json(data, path):

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
            default=json_safe,
        )


# =================================================================================================
# 8. STATUS MANAGEMENT
# =================================================================================================

def write_status(run_dir, state, extra=None):

    data = {
        "state": state,
        "timestamp": datetime.now().isoformat(),
    }

    if extra:
        data.update(extra)

    save_json(
        data,
        run_dir / "status.json",
    )


def read_status(run_dir):

    path = run_dir / "status.json"

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =================================================================================================
# 9. IMAGE VALIDATION / JUPYTER ARTIFACT EXCLUSION
# =================================================================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def valid_image_file(path_string):

    path = Path(path_string)

    # Prevent Jupyter artifacts entering the experiment.
    if ".ipynb_checkpoints" in path.parts:
        return False

    if any(
        part.startswith(".")
        for part in path.parts
        if part not in [".", ".."]
    ):
        # Hidden nested folders/files excluded.
        if ".ipynb_checkpoints" in path.parts:
            return False

    return path.suffix.lower() in VALID_EXTENSIONS


# =================================================================================================
# 10. MODEL CREATION
# =================================================================================================

def build_model():

    model = timm.create_model(
        TIMM_MODEL_NAME,
        pretrained=True,
        num_classes=EXPECTED_CLASSES,
    )

    return model


# =================================================================================================
# 11. CLEAN BASELINE PREPROCESSING
# =================================================================================================

def get_model_normalization(model):

    config = timm.data.resolve_model_data_config(model)

    mean = config.get(
        "mean",
        (0.485, 0.456, 0.406),
    )

    std = config.get(
        "std",
        (0.229, 0.224, 0.225),
    )

    return tuple(mean), tuple(std)


def make_clean_transform(mean, std):

    # Deterministic.
    #
    # No augmentation.
    #
    # Aspect ratio is preserved by Resize(shorter side = 256),
    # followed by deterministic center crop.

    return transforms.Compose(
        [
            transforms.Resize(
                RESIZE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),

            transforms.CenterCrop(
                IMAGE_SIZE
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


# =================================================================================================
# 12. DATASET CREATION
# =================================================================================================

def create_datasets_for_model(model):

    mean, std = get_model_normalization(model)

    transform = make_clean_transform(
        mean,
        std,
    )

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=transform,
        is_valid_file=valid_image_file,
        allow_empty=True,
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=transform,
        is_valid_file=valid_image_file,
        allow_empty=True,
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
        is_valid_file=valid_image_file,
        allow_empty=True,
    )

    if len(train_dataset.classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes in train, "
            f"found {len(train_dataset.classes)}."
        )

    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise RuntimeError(
            "Class-to-index mapping differs between train/val/test."
        )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        mean,
        std,
    )


# =================================================================================================
# 13. DATASET MANIFEST / FINGERPRINT SUPPORT
# =================================================================================================

def save_dataset_manifest(dataset, split_name, output_path):

    rows = []

    for image_path, class_index in dataset.samples:

        rows.append(
            {
                "split": split_name,
                "image_path": image_path,
                "class_index": int(class_index),
                "class_name": dataset.classes[class_index],
            }
        )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image_path",
                "class_index",
                "class_name",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


# =================================================================================================
# 14. DATA LOADERS
# =================================================================================================

def create_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    seed,
    generator_state=None,
):

    generator = torch.Generator()

    generator.manual_seed(seed)

    if generator_state is not None:
        generator.set_state(generator_state)

    common = {
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        generator,
    )


# =================================================================================================
# 15. PARAMETER / MODEL SIZE
# =================================================================================================

def parameter_counts(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# =================================================================================================
# 16. OPTIONAL FLOPs
# =================================================================================================

def estimate_flops(model):

    if not FVCORE_AVAILABLE:
        return None

    try:

        was_training = model.training

        model.eval()

        dummy = torch.randn(
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=DEVICE,
        )

        analysis = FlopCountAnalysis(
            model,
            dummy,
        )

        total_flops = float(
            analysis.total()
        )

        if was_training:
            model.train()

        return total_flops

    except Exception as e:

        print(
            f"FLOPs estimation skipped: {e}"
        )

        return None


# =================================================================================================
# 17. WARM-UP + COSINE SCHEDULER
# =================================================================================================

def create_scheduler(optimizer):

    def lr_lambda(epoch):

        if epoch < WARMUP_EPOCHS:

            return float(epoch + 1) / float(
                max(1, WARMUP_EPOCHS)
            )

        progress = (
            epoch - WARMUP_EPOCHS
        ) / float(
            max(
                1,
                EPOCHS - WARMUP_EPOCHS
            )
        )

        return 0.5 * (
            1.0 + math.cos(
                math.pi * progress
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


# =================================================================================================
# 18. TRAIN ONE EPOCH
# =================================================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
):

    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        if USE_AMP:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        batch_n = labels.size(0)

        loss_sum += (
            loss.item() * batch_n
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_n

    return (
        loss_sum / total,
        100.0 * correct / total,
    )


# =================================================================================================
# 19. VALIDATION
# =================================================================================================

@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    criterion,
):

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        n = labels.size(0)

        loss_sum += (
            loss.item() * n
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += n

    return (
        loss_sum / total,
        100.0 * correct / total,
    )


# =================================================================================================
# 20. TEST PREDICTIONS
# =================================================================================================

@torch.no_grad()
def collect_predictions(
    model,
    loader,
):

    model.eval()

    all_targets = []
    all_predictions = []
    all_confidences = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, predictions = (
            probs.max(dim=1)
        )

        all_targets.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_confidences.extend(
            confidence.cpu().numpy().tolist()
        )

    return (
        np.asarray(all_targets),
        np.asarray(all_predictions),
        np.asarray(all_confidences),
    )


# =================================================================================================
# 21. TEST METRICS
# =================================================================================================

def compute_metrics(
    targets,
    predictions,
):

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "test_top1_accuracy": float(
            accuracy * 100
        ),

        "test_macro_precision": float(
            precision * 100
        ),

        "test_macro_recall": float(
            recall * 100
        ),

        "test_macro_f1": float(
            f1 * 100
        ),
    }


# =================================================================================================
# 22. CLASSIFICATION REPORT
# =================================================================================================

def save_classification_report_csv(
    targets,
    predictions,
    classes,
    path,
):

    report = classification_report(
        targets,
        predictions,
        labels=list(
            range(len(classes))
        ),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_name in classes:

        x = report[class_name]

        rows.append(
            {
                "class": class_name,
                "precision": x["precision"],
                "recall": x["recall"],
                "f1_score": x["f1-score"],
                "support": x["support"],
            }
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "precision",
                "recall",
                "f1_score",
                "support",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


# =================================================================================================
# 23. CONFUSION MATRIX
# =================================================================================================

def save_confusion_matrix_csv(
    targets,
    predictions,
    classes,
    path,
):

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(len(classes))
        ),
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["true/pred"] + classes
        )

        for class_name, row in zip(
            classes,
            matrix,
        ):

            writer.writerow(
                [class_name]
                + row.tolist()
            )


# =================================================================================================
# 24. SAVE TEST PREDICTIONS
# =================================================================================================

def save_test_predictions(
    dataset,
    targets,
    predictions,
    confidences,
    path,
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "image_path",
                "true_index",
                "true_class",
                "predicted_index",
                "predicted_class",
                "confidence",
            ]
        )

        for (
            sample,
            target,
            prediction,
            confidence,
        ) in zip(
            dataset.samples,
            targets,
            predictions,
            confidences,
        ):

            writer.writerow(
                [
                    sample[0],
                    int(target),
                    dataset.classes[target],
                    int(prediction),
                    dataset.classes[prediction],
                    float(confidence),
                ]
            )


# =================================================================================================
# 25. INFERENCE BENCHMARK
# =================================================================================================

@torch.no_grad()
def benchmark_inference(model):

    model.eval()

    dummy = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    for _ in range(
        INFERENCE_WARMUP
    ):
        _ = model(dummy)

    torch.cuda.synchronize()

    timings = []

    for _ in range(
        INFERENCE_ITERATIONS
    ):

        torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(dummy)

        torch.cuda.synchronize()

        timings.append(
            time.perf_counter() - start
        )

    latency_mean = float(
        np.mean(timings)
    )

    latency_std = float(
        np.std(timings, ddof=1)
    )

    return {
        "batch_size": 1,

        "precision": "FP32",

        "warmup_iterations":
            INFERENCE_WARMUP,

        "timed_iterations":
            INFERENCE_ITERATIONS,

        "mean_latency_ms":
            latency_mean * 1000,

        "std_latency_ms":
            latency_std * 1000,

        "throughput_images_per_second":
            1.0 / latency_mean,
    }


# =================================================================================================
# 26. ENVIRONMENT METADATA
# =================================================================================================

def get_environment():

    props = torch.cuda.get_device_properties(
        0
    )

    return {
        "timestamp":
            datetime.now().isoformat(),

        "hostname":
            socket.gethostname(),

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "torchvision_version":
            torchvision.__version__,

        "timm_version":
            timm.__version__,

        "cuda_version":
            torch.version.cuda,

        "cudnn_version":
            torch.backends.cudnn.version(),

        "gpu_name":
            torch.cuda.get_device_name(0),

        "gpu_total_memory_gb":
            props.total_memory / (1024**3),

        "training_precision":
            "AMP-FP16" if USE_AMP else "FP32",

        "inference_benchmark_precision":
            "FP32",
    }


# =================================================================================================
# 27. RNG STATE SAVE / RESTORE
# =================================================================================================

def get_rng_state(generator):

    state = {
        "python_random_state":
            random.getstate(),

        "numpy_random_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),

        "loader_generator_state":
            generator.get_state(),
    }

    if torch.cuda.is_available():

        state[
            "cuda_rng_state_all"
        ] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(
    checkpoint,
    generator,
):

    if "python_random_state" in checkpoint:
        random.setstate(
            checkpoint[
                "python_random_state"
            ]
        )

    if "numpy_random_state" in checkpoint:
        np.random.set_state(
            checkpoint[
                "numpy_random_state"
            ]
        )

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint[
                "torch_rng_state"
            ]
        )

    if (
        "cuda_rng_state_all"
        in checkpoint
    ):

        torch.cuda.set_rng_state_all(
            checkpoint[
                "cuda_rng_state_all"
            ]
        )

    if (
        "loader_generator_state"
        in checkpoint
    ):

        generator.set_state(
            checkpoint[
                "loader_generator_state"
            ]
        )


# =================================================================================================
# 28. CHECKPOINT SAVE
# =================================================================================================

def save_last_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_accuracy,
    best_epoch,
    history,
    generator,
):

    rng = get_rng_state(
        generator
    )

    checkpoint = {
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_val_accuracy":
            best_val_accuracy,

        "best_epoch":
            best_epoch,

        "history":
            history,

        "model_key":
            MODEL,

        "timm_model":
            TIMM_MODEL_NAME,

        **rng,
    }

    torch.save(
        checkpoint,
        path,
    )


# =================================================================================================
# 29. TRAINING HISTORY CSV
# =================================================================================================

def save_history(history, path):

    if not history:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)


# =================================================================================================
# 30. ONE-SEED RUNNER
# =================================================================================================

def run_seed(seed):

    print("\n")
    print("=" * 100)
    print(
        f"{PAPER_MODEL_NAME} | SEED {seed}"
    )
    print("=" * 100)

    run_dir = (
        MODEL_RESULTS_DIR
        / f"seed_{seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = read_status(
        run_dir
    )

    if (
        status
        and status.get("state")
        == "COMPLETED"
        and (
            run_dir / "metrics.json"
        ).exists()
    ):

        print(
            f"Seed {seed} already completed. Skipping."
        )

        with open(
            run_dir / "metrics.json",
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    write_status(
        run_dir,
        "RUNNING",
        {
            "model":
                MODEL,

            "seed":
                seed,
        },
    )

    try:

        set_global_seed(
            seed
        )

        # -----------------------------------------------------------------------------------------
        # MODEL
        # -----------------------------------------------------------------------------------------

        model = build_model()

        (
            train_dataset,
            val_dataset,
            test_dataset,
            mean,
            std,
        ) = create_datasets_for_model(
            model
        )

        print(
            f"Train images : {len(train_dataset)}"
        )

        print(
            f"Val images   : {len(val_dataset)}"
        )

        print(
            f"Test images  : {len(test_dataset)}"
        )

        print(
            f"Classes      : {len(train_dataset.classes)}"
        )

        # -----------------------------------------------------------------------------------------
        # SAVE DATA MANIFEST ONCE PER MODEL
        # -----------------------------------------------------------------------------------------

        manifest_dir = (
            MODEL_RESULTS_DIR
            / "dataset_manifests"
        )

        manifest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not (
            manifest_dir
            / "train_manifest.csv"
        ).exists():

            save_dataset_manifest(
                train_dataset,
                "train",
                manifest_dir
                / "train_manifest.csv",
            )

            save_dataset_manifest(
                val_dataset,
                "val",
                manifest_dir
                / "val_manifest.csv",
            )

            save_dataset_manifest(
                test_dataset,
                "test",
                manifest_dir
                / "test_manifest.csv",
            )

        # -----------------------------------------------------------------------------------------
        # CHECKPOINT DETECTION
        # -----------------------------------------------------------------------------------------

        last_checkpoint_path = (
            run_dir
            / "checkpoint_last.pth"
        )

        resume_checkpoint = None

        generator_state = None

        if last_checkpoint_path.exists():

            print(
                "\nExisting checkpoint detected."
            )

            resume_checkpoint = torch.load(
                last_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            generator_state = (
                resume_checkpoint.get(
                    "loader_generator_state"
                )
            )

        # -----------------------------------------------------------------------------------------
        # LOADERS
        # -----------------------------------------------------------------------------------------

        (
            train_loader,
            val_loader,
            test_loader,
            loader_generator,
        ) = create_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            seed,
            generator_state,
        )

        # -----------------------------------------------------------------------------------------
        # MODEL TO GPU
        # -----------------------------------------------------------------------------------------

        model = model.to(
            DEVICE
        )

        total_params, trainable_params = (
            parameter_counts(model)
        )

        # FLOPs calculated once per seed;
        # retained for reproducibility even though architecture is identical between seeds.

        flops = estimate_flops(
            model
        )

        # -----------------------------------------------------------------------------------------
        # LOSS / OPTIMIZER / SCHEDULER
        # -----------------------------------------------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = create_scheduler(
            optimizer
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=USE_AMP,
        )

        # -----------------------------------------------------------------------------------------
        # RUN CONFIG
        # -----------------------------------------------------------------------------------------

        run_config = {
            "experiment":
                "LODO-3_CFRT_Images2021_to_UAlpha40",
                

            "model_key":
                MODEL,

            "paper_model_name":
                PAPER_MODEL_NAME,

            "timm_model_name":
                TIMM_MODEL_NAME,

            "pretrained":
                True,

            "seed":
                seed,

            "dataset_root":
                str(DATA_ROOT),

            "train_images":
                len(train_dataset),

            "val_images":
                len(val_dataset),

            "test_images":
                len(test_dataset),

            "number_of_classes":
                len(train_dataset.classes),

            "classes":
                train_dataset.classes,

            "class_to_idx":
                train_dataset.class_to_idx,

            "epochs":
                EPOCHS,

            "warmup_epochs":
                WARMUP_EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "num_workers":
                NUM_WORKERS,

            "input_size":
                IMAGE_SIZE,

            "resize_size":
                RESIZE_SIZE,

            "preprocessing":
                "Resize(256)-CenterCrop(224)-ToTensor-Normalize",

            "normalization_mean":
                mean,

            "normalization_std":
                std,

            "augmentation":
                False,

            "mixup":
                False,

            "cutmix":
                False,

            "tta":
                False,

            "artificial_perturbation":
                False,

            "optimizer":
                "AdamW",

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "scheduler":
                "5-epoch linear warmup + cosine decay",

            "loss":
                "CrossEntropyLoss",

            "early_stopping":
                False,

            "checkpoint_selection":
                "highest validation accuracy",

            "training_precision":
                "AMP-FP16"
                if USE_AMP
                else "FP32",

            "total_parameters":
                total_params,

            "trainable_parameters":
                trainable_params,

            "estimated_flops":
                flops,

            "environment":
                get_environment(),
        }

        save_json(
            run_config,
            run_dir / "run_config.json",
        )

        # -----------------------------------------------------------------------------------------
        # RESUME
        # -----------------------------------------------------------------------------------------

        start_epoch = 1

        best_val_accuracy = -1.0
        best_epoch = 0

        history = []

        accumulated_training_seconds = 0.0

        if resume_checkpoint is not None:

            model.load_state_dict(
                resume_checkpoint[
                    "model_state_dict"
                ]
            )

            optimizer.load_state_dict(
                resume_checkpoint[
                    "optimizer_state_dict"
                ]
            )

            scheduler.load_state_dict(
                resume_checkpoint[
                    "scheduler_state_dict"
                ]
            )

            scaler.load_state_dict(
                resume_checkpoint[
                    "scaler_state_dict"
                ]
            )

            best_val_accuracy = (
                resume_checkpoint[
                    "best_val_accuracy"
                ]
            )

            best_epoch = (
                resume_checkpoint[
                    "best_epoch"
                ]
            )

            history = (
                resume_checkpoint.get(
                    "history",
                    [],
                )
            )

            if history:

                accumulated_training_seconds = sum(
                    x[
                        "epoch_time_seconds"
                    ]
                    for x in history
                )

            start_epoch = (
                resume_checkpoint[
                    "epoch"
                ] + 1
            )

            restore_rng_state(
                resume_checkpoint,
                loader_generator,
            )

            print(
                f"Resuming seed {seed} from epoch {start_epoch}."
            )

        else:

            print(
                f"Starting seed {seed} from epoch 1."
            )

        # -----------------------------------------------------------------------------------------
        # GPU MEMORY
        # -----------------------------------------------------------------------------------------

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        # -----------------------------------------------------------------------------------------
        # TRAIN
        # -----------------------------------------------------------------------------------------

        for epoch in range(
            start_epoch,
            EPOCHS + 1,
        ):

            epoch_start = (
                time.perf_counter()
            )

            train_loss, train_acc = (
                train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scaler,
                )
            )

            val_loss, val_acc = (
                evaluate_loader(
                    model,
                    val_loader,
                    criterion,
                )
            )

            epoch_time = (
                time.perf_counter()
                - epoch_start
            )

            current_lr = (
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            row = {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_acc,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_acc,

                "learning_rate":
                    current_lr,

                "epoch_time_seconds":
                    epoch_time,
            }

            history.append(
                row
            )

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.2f}% | "
                f"LR {current_lr:.8f} | "
                f"{epoch_time:.1f}s"
            )

            # -------------------------------------------------------------------------------------
            # BEST MODEL
            # -------------------------------------------------------------------------------------

            if val_acc > best_val_accuracy:

                best_val_accuracy = (
                    val_acc
                )

                best_epoch = epoch

                torch.save(
                    {
                        "epoch":
                            epoch,

                        "model_state_dict":
                            model.state_dict(),

                        "val_accuracy":
                            val_acc,

                        "model":
                            MODEL,

                        "seed":
                            seed,

                        "classes":
                            train_dataset.classes,
                    },

                    run_dir
                    / "best_model.pth",
                )

            # -------------------------------------------------------------------------------------
            # SCHEDULER
            # -------------------------------------------------------------------------------------

            scheduler.step()

            # -------------------------------------------------------------------------------------
            # HISTORY
            # -------------------------------------------------------------------------------------

            save_history(
                history,
                run_dir
                / "training_history.csv",
            )

            # -------------------------------------------------------------------------------------
            # RESUMABLE CHECKPOINT
            # -------------------------------------------------------------------------------------

            save_last_checkpoint(
                last_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_accuracy,
                best_epoch,
                history,
                loader_generator,
            )

            write_status(
                run_dir,
                "RUNNING",
                {
                    "model":
                        MODEL,

                    "seed":
                        seed,

                    "last_completed_epoch":
                        epoch,

                    "best_epoch":
                        best_epoch,

                    "best_val_accuracy":
                        best_val_accuracy,
                },
            )

        # -----------------------------------------------------------------------------------------
        # TRAINING TIME
        # -----------------------------------------------------------------------------------------

        total_training_seconds = sum(
            x[
                "epoch_time_seconds"
            ]
            for x in history
        )

        mean_epoch_time = float(
            np.mean(
                [
                    x[
                        "epoch_time_seconds"
                    ]
                    for x in history
                ]
            )
        )

        # -----------------------------------------------------------------------------------------
        # GPU MEMORY
        # -----------------------------------------------------------------------------------------

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated()
            / (1024**3)
        )

        peak_reserved_gb = (
            torch.cuda.max_memory_reserved()
            / (1024**3)
        )

        # -----------------------------------------------------------------------------------------
        # LOAD BEST MODEL
        # -----------------------------------------------------------------------------------------

        best_checkpoint = torch.load(
            run_dir
            / "best_model.pth",
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            best_checkpoint[
                "model_state_dict"
            ]
        )

        # -----------------------------------------------------------------------------------------
        # TEST
        # -----------------------------------------------------------------------------------------

        (
            targets,
            predictions,
            confidences,
        ) = collect_predictions(
            model,
            test_loader,
        )

        metrics = compute_metrics(
            targets,
            predictions,
        )

        # -----------------------------------------------------------------------------------------
        # DETAILED TEST OUTPUT
        # -----------------------------------------------------------------------------------------

        save_test_predictions(
            test_dataset,
            targets,
            predictions,
            confidences,
            run_dir
            / "test_predictions.csv",
        )

        save_classification_report_csv(
            targets,
            predictions,
            train_dataset.classes,
            run_dir
            / "classification_report.csv",
        )

        save_confusion_matrix_csv(
            targets,
            predictions,
            train_dataset.classes,
            run_dir
            / "confusion_matrix.csv",
        )

        # -----------------------------------------------------------------------------------------
        # INFERENCE SPEED
        # -----------------------------------------------------------------------------------------

        inference = benchmark_inference(
            model
        )

        # -----------------------------------------------------------------------------------------
        # CHECKPOINT SIZE
        # -----------------------------------------------------------------------------------------

        model_size_mb = (
            (
                run_dir
                / "best_model.pth"
            ).stat().st_size
            / (1024**2)
        )

        # -----------------------------------------------------------------------------------------
        # FINAL METRICS
        # -----------------------------------------------------------------------------------------

        final_metrics = {

            "model_key":
                MODEL,

            "paper_model_name":
                PAPER_MODEL_NAME,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "best_val_accuracy":
                best_val_accuracy,

            **metrics,

            "epochs_completed":
                len(history),

            "training_time_seconds":
                total_training_seconds,

            "training_time_hours":
                total_training_seconds
                / 3600.0,

            "mean_epoch_time_seconds":
                mean_epoch_time,

            "total_parameters":
                total_params,

            "trainable_parameters":
                trainable_params,

            "estimated_flops":
                flops,

            "peak_gpu_memory_allocated_gb":
                peak_allocated_gb,

            "peak_gpu_memory_reserved_gb":
                peak_reserved_gb,

            "best_checkpoint_size_mb":
                model_size_mb,

            "inference_mean_latency_ms":
                inference[
                    "mean_latency_ms"
                ],

            "inference_std_latency_ms":
                inference[
                    "std_latency_ms"
                ],

            "inference_throughput_images_per_second":
                inference[
                    "throughput_images_per_second"
                ],

            "inference_batch_size":
                1,

            "inference_precision":
                "FP32",

            "augmentation":
                False,

            "tta":
                False,

            "perturbation":
                False,
        }

        save_json(
            final_metrics,
            run_dir
            / "metrics.json",
        )

        # -----------------------------------------------------------------------------------------
        # HUMAN READABLE PAPER NOTES
        # -----------------------------------------------------------------------------------------

        with open(
            run_dir
            / "paper_run_summary.txt",
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                f"Model: {PAPER_MODEL_NAME}\n"
            )

            f.write(
                f"Exact timm model: {TIMM_MODEL_NAME}\n"
            )

            f.write(
                f"Seed: {seed}\n"
            )

            f.write(
                f"Best epoch: {best_epoch}\n"
            )

            f.write(
                f"Best validation accuracy: "
                f"{best_val_accuracy:.4f}%\n"
            )

            f.write(
                f"Test Top-1 accuracy: "
                f"{metrics['test_top1_accuracy']:.4f}%\n"
            )

            f.write(
                f"Macro Precision: "
                f"{metrics['test_macro_precision']:.4f}%\n"
            )

            f.write(
                f"Macro Recall: "
                f"{metrics['test_macro_recall']:.4f}%\n"
            )

            f.write(
                f"Macro F1: "
                f"{metrics['test_macro_f1']:.4f}%\n"
            )

            f.write(
                f"Parameters: "
                f"{total_params:,}\n"
            )

            if flops is not None:

                f.write(
                    f"Estimated FLOPs: "
                    f"{flops:,.0f}\n"
                )

            f.write(
                f"Training time (hours): "
                f"{total_training_seconds / 3600:.4f}\n"
            )

            f.write(
                f"Peak allocated GPU memory (GB): "
                f"{peak_allocated_gb:.4f}\n"
            )

            f.write(
                f"Peak reserved GPU memory (GB): "
                f"{peak_reserved_gb:.4f}\n"
            )

            f.write(
                f"FP32 batch-1 inference latency (ms): "
                f"{inference['mean_latency_ms']:.4f}\n"
            )

            f.write(
                f"FP32 batch-1 throughput (images/s): "
                f"{inference['throughput_images_per_second']:.4f}\n"
            )

            f.write(
                "Training augmentation: No\n"
            )

            f.write(
                "TTA: No\n"
            )

            f.write(
                "Artificial perturbation during training: No\n"
            )

        # -----------------------------------------------------------------------------------------
        # COMPLETED
        # -----------------------------------------------------------------------------------------

        write_status(
            run_dir,
            "COMPLETED",
            {
                "model":
                    MODEL,

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "best_val_accuracy":
                    best_val_accuracy,

                "test_top1_accuracy":
                    metrics[
                        "test_top1_accuracy"
                    ],

                "test_macro_f1":
                    metrics[
                        "test_macro_f1"
                    ],
            },
        )

        print("\nSeed completed:")
        print(
            f"  Best Val Acc : {best_val_accuracy:.2f}%"
        )
        print(
            f"  Test Top-1   : {metrics['test_top1_accuracy']:.2f}%"
        )
        print(
            f"  Macro F1     : {metrics['test_macro_f1']:.2f}%"
        )
        print(
            f"  Peak GPU     : {peak_allocated_gb:.3f} GB"
        )
        print(
            f"  Latency B1   : {inference['mean_latency_ms']:.3f} ms"
        )

        torch.cuda.empty_cache()

        return final_metrics

    except Exception as e:

        write_status(
            run_dir,
            "FAILED",
            {
                "model":
                    MODEL,

                "seed":
                    seed,

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc(),
            },
        )

        print(
            "\nRUN FAILED."
        )

        print(
            traceback.format_exc()
        )

        raise


# =================================================================================================
# 31. CONFIDENCE INTERVAL
# =================================================================================================

def mean_sd_ci(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    mean = float(
        np.mean(values)
    )

    if n < 2:

        return {
            "n": n,
            "mean": mean,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    sd = float(
        np.std(
            values,
            ddof=1,
        )
    )

    se = sd / math.sqrt(n)

    if SCIPY_AVAILABLE:

        critical = float(
            student_t.ppf(
                0.975,
                df=n - 1,
            )
        )

        ci_method = (
            "Student-t 95% CI"
        )

    else:

        critical = 1.96

        ci_method = (
            "Normal approximation 95% CI"
        )

    margin = critical * se

    return {
        "n":
            n,

        "mean":
            mean,

        "sd":
            sd,

        "ci95_low":
            mean - margin,

        "ci95_high":
            mean + margin,

        "ci_method":
            ci_method,
    }


# =================================================================================================
# 32. MODEL-LEVEL AGGREGATION
# =================================================================================================

def aggregate_model_results():

    rows = []

    for seed in SEEDS:

        metrics_path = (
            MODEL_RESULTS_DIR
            / f"seed_{seed}"
            / "metrics.json"
        )

        if not metrics_path.exists():
            continue

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as f:

            rows.append(
                json.load(f)
            )

    if not rows:

        print(
            "No completed seed results available."
        )

        return

    # ---------------------------------------------------------------------------------------------
    # SAVE PER-SEED SUMMARY
    # ---------------------------------------------------------------------------------------------

    summary_csv = (
        MODEL_RESULTS_DIR
        / "seed_results.csv"
    )

    preferred_fields = [
        "seed",
        "best_epoch",
        "best_val_accuracy",
        "test_top1_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "training_time_hours",
        "peak_gpu_memory_allocated_gb",
        "peak_gpu_memory_reserved_gb",
        "inference_mean_latency_ms",
        "inference_throughput_images_per_second",
        "total_parameters",
        "estimated_flops",
        "best_checkpoint_size_mb",
    ]

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=preferred_fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        row.get(key)
                    for key
                    in preferred_fields
                }
            )

    # ---------------------------------------------------------------------------------------------
    # AGGREGATED METRICS
    # ---------------------------------------------------------------------------------------------

    aggregate_fields = {

        "test_top1_accuracy":
            "Top-1 Accuracy (%)",

        "test_macro_precision":
            "Macro Precision (%)",

        "test_macro_recall":
            "Macro Recall (%)",

        "test_macro_f1":
            "Macro F1 (%)",

        "best_val_accuracy":
            "Best Validation Accuracy (%)",

        "training_time_hours":
            "Training Time (hours)",

        "peak_gpu_memory_allocated_gb":
            "Peak Allocated GPU Memory (GB)",

        "inference_mean_latency_ms":
            "Inference Latency B1 (ms)",

        "inference_throughput_images_per_second":
            "Inference Throughput B1 (images/s)",
    }

    aggregate = {

        "model_key":
            MODEL,

        "paper_model_name":
            PAPER_MODEL_NAME,

        "timm_model_name":
            TIMM_MODEL_NAME,

        "completed_seeds":
            [
                row[
                    "seed"
                ]
                for row in rows
            ],

        "number_of_completed_seeds":
            len(rows),

        "total_parameters":
            rows[0].get(
                "total_parameters"
            ),

        "estimated_flops":
            rows[0].get(
                "estimated_flops"
            ),

        "metrics":
            {},
    }

    for field, label in (
        aggregate_fields.items()
    ):

        values = [
            row[field]
            for row in rows
            if row.get(field)
            is not None
        ]

        aggregate[
            "metrics"
        ][field] = {
            "label":
                label,

            **mean_sd_ci(
                values
            ),
        }

    save_json(
        aggregate,
        MODEL_RESULTS_DIR
        / "model_summary.json",
    )

    # ---------------------------------------------------------------------------------------------
    # PAPER-READY CSV
    # ---------------------------------------------------------------------------------------------

    paper_rows = []

    for key, info in (
        aggregate["metrics"].items()
    ):

        paper_rows.append(
            {
                "metric":
                    info[
                        "label"
                    ],

                "n":
                    info[
                        "n"
                    ],

                "mean":
                    info[
                        "mean"
                    ],

                "sd":
                    info[
                        "sd"
                    ],

                "ci95_low":
                    info[
                        "ci95_low"
                    ],

                "ci95_high":
                    info[
                        "ci95_high"
                    ],
            }
        )

    with open(
        MODEL_RESULTS_DIR
        / "paper_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "n",
                "mean",
                "sd",
                "ci95_low",
                "ci95_high",
            ],
        )

        writer.writeheader()
        writer.writerows(
            paper_rows
        )

    # ---------------------------------------------------------------------------------------------
    # HUMAN-READABLE PAPER SUMMARY
    # ---------------------------------------------------------------------------------------------

    with open(
        MODEL_RESULTS_DIR
        / "paper_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PSL MAJOR REVISION — STRATIFIED BASELINE\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"Model: {PAPER_MODEL_NAME}\n"
        )

        f.write(
            f"Exact timm identifier: {TIMM_MODEL_NAME}\n"
        )

        f.write(
            f"Dataset: {DATA_ROOT}\n"
        )

        f.write(
            f"Completed seeds: "
            f"{aggregate['completed_seeds']}\n"
        )

        f.write(
            f"Number of runs: "
            f"{len(rows)}\n\n"
        )

        for key, info in (
            aggregate[
                "metrics"
            ].items()
        ):

            if info["sd"] is None:

                f.write(
                    f"{info['label']}: "
                    f"{info['mean']:.4f}\n"
                )

            else:

                f.write(
                    f"{info['label']}: "
                    f"{info['mean']:.4f} "
                    f"± {info['sd']:.4f}; "
                    f"95% CI "
                    f"[{info['ci95_low']:.4f}, "
                    f"{info['ci95_high']:.4f}]\n"
                )

        f.write("\n")
        f.write(
            f"Parameters: "
            f"{rows[0]['total_parameters']:,}\n"
        )

        if (
            rows[0].get(
                "estimated_flops"
            )
            is not None
        ):

            f.write(
                f"Estimated FLOPs: "
                f"{rows[0]['estimated_flops']:,.0f}\n"
            )

        f.write(
            "\nBaseline configuration:\n"
        )

        f.write(
            "- No data augmentation\n"
        )

        f.write(
            "- No MixUp\n"
        )

        f.write(
            "- No CutMix\n"
        )

        f.write(
            "- No TTA\n"
        )

        f.write(
            "- No artificial perturbations during training\n"
        )

        f.write(
            "- Deterministic Resize + CenterCrop + normalization\n"
        )

        f.write(
            "- Best checkpoint selected using validation accuracy\n"
        )

        f.write(
            "- Test set used only for final evaluation\n"
        )

    # ---------------------------------------------------------------------------------------------
    # DISPLAY SUMMARY
    # ---------------------------------------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print(
        f"FINAL MODEL SUMMARY — {PAPER_MODEL_NAME}"
    )
    print("=" * 100)

    print(
        f"Completed seeds: "
        f"{aggregate['completed_seeds']}"
    )

    for field in [
        "test_top1_accuracy",
        "test_macro_f1",
        "training_time_hours",
        "peak_gpu_memory_allocated_gb",
        "inference_mean_latency_ms",
    ]:

        info = aggregate[
            "metrics"
        ][field]

        if info["sd"] is None:

            print(
                f"{info['label']}: "
                f"{info['mean']:.4f}"
            )

        else:

            print(
                f"{info['label']}: "
                f"{info['mean']:.4f} "
                f"± {info['sd']:.4f} "
                f"| 95% CI "
                f"[{info['ci95_low']:.4f}, "
                f"{info['ci95_high']:.4f}]"
            )

    print(
        f"\nResults saved to:\n"
        f"{MODEL_RESULTS_DIR}"
    )

    print("=" * 100)


# =================================================================================================
# 33. PRE-FLIGHT REPORT
# =================================================================================================

print("\n")
print("=" * 100)
print("PSL MAJOR REVISION — MASTER TRAINER PRE-FLIGHT")
print("=" * 100)

print(
    f"Model                    : {PAPER_MODEL_NAME}"
)

print(
    f"timm identifier          : {TIMM_MODEL_NAME}"
)

print(
    f"Dataset                  : {DATA_ROOT}"
)

print(
    f"Output                   : {MODEL_RESULTS_DIR}"
)

print(
    f"Seeds                    : {SEEDS}"
)

print(
    f"Epochs per seed          : {EPOCHS}"
)

print(
    f"Warm-up epochs           : {WARMUP_EPOCHS}"
)

print(
    f"Batch size               : {BATCH_SIZE}"
)

print(
    f"Training precision       : "
    f"{'AMP-FP16' if USE_AMP else 'FP32'}"
)

print(
    f"GPU                      : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"CUDA                     : "
    f"{torch.version.cuda}"
)

print(
    f"PyTorch                  : "
    f"{torch.__version__}"
)

print(
    f"timm                     : "
    f"{timm.__version__}"
)

print(
    f"fvcore FLOPs available   : "
    f"{FVCORE_AVAILABLE}"
)

print(
    "Training augmentation    : NO"
)

print(
    "TTA                      : NO"
)

print(
    "Training perturbations   : NO"
)

print(
    "Resume support           : YES — epoch + seed level"
)

print("=" * 100)


# =================================================================================================
# 34. RUN ALL SEEDS AUTOMATICALLY
# =================================================================================================

all_seed_results = []

for seed in SEEDS:

    result = run_seed(
        seed
    )

    if result is not None:
        all_seed_results.append(
            result
        )


# =================================================================================================
# 35. AGGREGATE ALL COMPLETED RUNS
# =================================================================================================

aggregate_model_results()