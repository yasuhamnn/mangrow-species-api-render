"""
Mangrow Species API — PyTorch inference using Kaggle-trained EfficientNet-B0 (timm).
No Keras/TensorFlow needed.

Place next to this file (from Kaggle /kaggle/working):
  - mangrow_efficientnetb0_best.pt
  - labels.json

Then run:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health
  POST /predict   (multipart field: image)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PT_PATH = ROOT / "mangrow_efficientnetb0_best.pt"
LABELS_PATH = ROOT / "labels.json"

if not PT_PATH.is_file() or not LABELS_PATH.is_file():
    raise FileNotFoundError(
        "Missing mangrow_efficientnetb0_best.pt or labels.json in ai-server/.\n"
        "Download them from Kaggle /kaggle/working and place them next to main.py."
    )

labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
CLASS_BY_IDX = {int(c["index"]): c for c in labels["classes"]}
IMG_SIZE = int(labels["input_size"])
NUM_CLASSES = int(labels["num_classes"])
TOP_K = 1
THR = float(labels.get("uncertain_threshold", 0.25))
# Unknown / out-of-distribution thresholds.
# Tuned so that a CORRECT mangrove prediction at ~40%+ confidence is NOT flagged unknown.
# Unknown is triggered only on STRONG evidence of OOD:
#   (a) extremely low top-1 confidence AND very spread-out probs, OR
#   (b) extremely high entropy across classes.
UNKNOWN_MAX_PROB_THR = float(labels.get("unknown_max_prob_threshold", 0.30))
UNKNOWN_ENTROPY_THR = float(labels.get("unknown_entropy_threshold", 1.85))
UNKNOWN_MARGIN_THR = float(labels.get("unknown_margin_threshold", 0.03))
# If top-1 confidence is above this value, we treat it as "confident enough" for a real in-class
# prediction (because your post-TTA held-out F1 is ~99% — the model's honest low ~40% calibration
# on novel real-world captures is still correct, so don't punish it).
IN_DISTRIBUTION_MAX_PROB_CONFIDENT = float(labels.get("in_distribution_max_prob_confident", 0.40))
USE_TTA = True
TTA_WEIGHTS = [0.50, 0.25, 0.25]  # center > hflip > 1.12x crop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PAD_VALUE = (114, 114, 114)


# ----------------------------------------------------------------------
# Model architecture — MUST match Kaggle training definition exactly
# ----------------------------------------------------------------------
class MangroveEfficientNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.35):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Install timm: pip install timm>=1.0.0") from e
        self.backbone = timm.create_model(
            "efficientnet_b0.ra_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            drop_rate=0.10,
            drop_path_rate=0.10,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.75),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _load_model() -> MangroveEfficientNet:
    model = MangroveEfficientNet(num_classes=NUM_CLASSES, dropout=0.35)
    ckpt = torch.load(str(PT_PATH), map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading {PT_PATH.name}: {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading {PT_PATH.name}: {unexpected[:5]}")
    model.to(DEVICE).eval()
    return model


MODEL = _load_model()


# ----------------------------------------------------------------------
# Preprocessing — mirrors Kaggle val_transform exactly:
#   LongestMaxSize -> PadIfNeeded (constant 114,114,114) -> CenterCrop -> Normalize
# ----------------------------------------------------------------------
def _longest_max_size_pad(img: Image.Image, target_long: int) -> Image.Image:
    w, h = img.size
    scale = target_long / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_long, target_long), PAD_VALUE)
    canvas.paste(img, ((target_long - new_w) // 2, (target_long - new_h) // 2))
    return canvas


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _normalize_tensor(np_img: np.ndarray) -> torch.Tensor:
    x = np_img.astype(np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))
    t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
    return t


def preprocess(img: Image.Image, view: str = "center") -> torch.Tensor:
    """Supported views: center | hflip | zoom112"""
    if view == "hflip":
        img = ImageOps.mirror(img)
    if view == "zoom112":
        target_long = int(IMG_SIZE * 1.12)
    else:
        target_long = IMG_SIZE
    x = _longest_max_size_pad(img, target_long)
    x = _center_crop(x, IMG_SIZE)
    arr = np.asarray(x.convert("RGB"), dtype=np.uint8)
    return _normalize_tensor(arr)


@torch.no_grad()
def predict_probs(img: Image.Image) -> np.ndarray:
    views: List[Tuple[str, float]]
    if USE_TTA:
        views = [("center", TTA_WEIGHTS[0]), ("hflip", TTA_WEIGHTS[1]), ("zoom112", TTA_WEIGHTS[2])]
    else:
        views = [("center", 1.0)]

    tensors = [preprocess(img, v) for v, _w in views]
    x = torch.cat(tensors, dim=0)
    logits = MODEL(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()  # shape [views, NUM_CLASSES]
    weights = np.array([w for _v, w in views], dtype=np.float32).reshape(-1, 1)
    weighted = (probs * weights).sum(axis=0) / float(weights.sum())
    return weighted.astype(np.float32)


def open_image_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # Fix EXIF orientation so phone photos (portrait/landscape) are upright:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def _classify_unknown(probs: np.ndarray) -> dict:
    """Decide whether an image is out-of-distribution / not one of the 13 trained species.

    Conservative rule: we ONLY flag unknown on strong evidence, because a "correct but ~40%
    confident" in-distribution prediction is common on real-world phone captures and we MUST
    NOT show the red "not part of 13 species" banner for them.

    Conditions that mark UNKNOWN (any):
      1. top-1 < UNKNOWN_MAX_PROB_THR AND entropy >= UNKNOWN_ENTROPY_THR AND margin < UNKNOWN_MARGIN_THR
      2. entropy is EXTREMELY high (normalized >= 0.72), regardless of raw max-prob
    Conditions that force KNOWN (safe override):
      A. top-1 >= IN_DISTRIBUTION_MAX_PROB_CONFIDENT, OR
      B. margin is big (> 0.18), even if entropy is somewhat high

    Returns a dict with flags + diagnostic values for the UI.
    """
    eps = 1e-12
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    p = p / float(p.sum())
    order = np.argsort(-p)
    top_1 = int(order[0])
    top_2 = int(order[1]) if len(order) > 1 else top_1
    max_prob = float(p[top_1])
    second_prob = float(p[top_2])
    margin = float(max_prob - second_prob)
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(max(2, len(p))))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Known-safe overrides first (most important for your 40%-confident correct IDs)
    clearly_in_distribution = bool(
        max_prob >= IN_DISTRIBUTION_MAX_PROB_CONFIDENT or margin >= 0.18
    )

    flags = {
        "low_max_prob": bool(max_prob < UNKNOWN_MAX_PROB_THR),
        "high_entropy": bool(entropy >= UNKNOWN_ENTROPY_THR or norm_entropy >= 0.58),
        "small_margin": bool(margin < UNKNOWN_MARGIN_THR),
        "extreme_entropy": bool(norm_entropy >= 0.72),
    }
    reasons = []
    # Unknown triggers
    if flags["low_max_prob"] and flags["high_entropy"] and flags["small_margin"]:
        reasons.append("very_low_confidence_spread_across_classes")
    if flags["extreme_entropy"]:
        reasons.append("predictions_very_uniform_across_13_classes")

    is_unknown = bool(
        not clearly_in_distribution
        and (len(reasons) > 0)
    )
    return {
        "is_unknown": is_unknown,
        "reasons": reasons,
        "top_1_idx": top_1,
        "top_2_idx": top_2,
        "top_1_confidence": max_prob,
        "top_2_confidence": second_prob,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
    }


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
app = FastAPI(title="Mangrow Species API (PyTorch)", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "framework": "pytorch+timm",
        "device": str(DEVICE),
        "model": labels.get("model"),
        "num_classes": NUM_CLASSES,
        "input_size": IMG_SIZE,
        "tta": USE_TTA,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Send an image multipart field named 'image'")
    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        img = open_image_rgb(data)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    try:
        probs = predict_probs(img)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}") from e

    ood = _classify_unknown(probs)
    top_idx = np.argsort(-probs)[:TOP_K]
    predictions = []
    for i in top_idx.tolist():
        meta = CLASS_BY_IDX[int(i)]
        predictions.append(
            {
                "db_name": meta["db_name"],
                "scientific_name": meta["scientific_name"],
                "common_name": meta.get("common_name"),
                "confidence": float(probs[int(i)]),
            }
        )
    top_conf = float(predictions[0]["confidence"])
    # If OOD: still return the top guess (for reference), but mark unknown explicitly.
    uncertain = bool(top_conf < THR)
    not_in_13_species = bool(ood["is_unknown"])
    return {
        "predictions": predictions,
        "uncertain": uncertain,
        "not_in_13_species": not_in_13_species,
        "ood_diagnostics": {
            "reasons": ood["reasons"],
            "top_1_confidence": ood["top_1_confidence"],
            "top_2_confidence": ood["top_2_confidence"],
            "margin": ood["margin"],
            "entropy": ood["entropy"],
            "normalized_entropy": ood["normalized_entropy"],
            "thresholds": {
                "unknown_max_prob": UNKNOWN_MAX_PROB_THR,
                "unknown_entropy": UNKNOWN_ENTROPY_THR,
                "unknown_margin": UNKNOWN_MARGIN_THR,
                "uncertain": THR,
            },
        },
    }
