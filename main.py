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

import base64
import io
import json
import os
from pathlib import Path
from typing import List, Tuple

import httpx
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

# ----------------------------------------------------------------------
# Roboflow health/disease inference proxy.
# The secret API key is stored ONLY on the server, never in the Expo client/APK.
# ----------------------------------------------------------------------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_HEALTH_MODEL = os.environ.get(
    "ROBOFLOW_HEALTH_MODEL", "mangrove_species_diseases-tnqrq/1"
).strip()
ROBOFLOW_BASE_URL = os.environ.get(
    "ROBOFLOW_BASE_URL", "https://detect.roboflow.com"
).strip().rstrip("/")
ROBOFLOW_TIMEOUT_S = float(os.environ.get("ROBOFLOW_TIMEOUT_S", "20.0"))
# Thresholds for Roboflow result normalization.
ROBOFLOW_UNCERTAIN_CONF = float(os.environ.get("ROBOFLOW_UNCERTAIN_CONF", "0.55"))
# Class-name keywords that indicate a disease/stress state (otherwise Healthy).
# Tuned to the exact public Universe "Mangrove_Species_Diseases" Object Detection model
# (12 classes: Black_spots, Brown_spots, Dieback-Gall + several mangrove species classes).
ROBOFLOW_STRESS_KEYWORDS: tuple[str, ...] = (
    "disease",
    "sick",
    "stress",
    "unhealthy",
    "blight",
    "spot",
    "spots",
    "rot",
    "rust",
    "mildew",
    "infected",
    "defect",
    "damage",
    "scorch",
    "scab",
    "mangrove_canker",
    "leaf_spot",
    "dieback",
    # Exact project disease classes (forked dataset is now 3 classes):
    "black_spots",
    "brown_spots",
    "white_spots",
    "dieback-gall",
    "dieback_gall",
    "black spots",
    "brown spots",
    "white spots",
    "dieback gall",
)
ROBOFLOW_DISEASE_CLASSES_EXACT: tuple[str, ...] = (
    # Any class name that equals one of these (case-insensitive) is disease,
    # regardless of keyword match, so UI stays accurate to the actual project classes.
    "black_spots",
    "brown_spots",
    "white_spots",
    "dieback-gall",
    "dieback_gall",
    "black spots",
    "brown spots",
    "white spots",
    "dieback gall",
)
ROBOFLOW_HEALTHY_KEYWORDS: tuple[str, ...] = (
    "healthy",
    "normal",
    "no_disease",
    "good",
    "sound",
)
# Non-disease classes in the universe project are mangrove species (not the leaf itself
# being "unhealthy"), so for those we mark them as "neutral" -> UI treats as healthy-ish
# but still flags "review by LGU" if no explicit healthy class was detected.
ROBOFLOW_NEUTRAL_CLASSES_EXACT: tuple[str, ...] = (
    "lumnitzera-littorea",
    "lumnitzera_littorea",
    "lumnitzera littorea",
    "lumnitzera-littorea-flower",
    "lumnitzera_littorea_flower",
    "lumnitzera littorea flower",
    "rhizophora-apiculata",
    "rhizophora_apiculata",
    "rhizophora apiculata",
    "rhizophora-apiculata-propagule",
    "rhizophora_apiculata_propagule",
    "rhizophora apiculata propagule",
    "scyphiphora-hydrophyl lacea",
    "scyphiphora-hydrophyl lacea-flower",
    "scyphiphora-hydrophyllacea",
    "scyphiphora_hydrophyllacea",
    "scyphiphora hydrophyllacea",
    "scyphiphora-hydrophyllacea-flower",
    "scyphiphora_hydrophyllacea_flower",
    "scyphiphora hydrophyllacea flower",
    "sonneratia-alba",
    "sonneratia_alba",
    "sonneratia alba",
    "sonneratia alba flower",
    "sonneratia-alba-flower",
    "sonneratia_alba_flower",
    "avicennia-alba",
    "avicennia_alba",
    "avicennia alba",
    "avicennia-marina",
    "avicennia_marina",
    "avicennia marina",
    "rhizophora-mucronata",
    "rhizophora_mucronata",
    "rhizophora mucronata",
    "rhizophora-mucronata-propagule",
    "bruguiera-sexangula",
    "bruguiera_sexangula",
    "bruguiera sexangula",
    "bruguiera-parviflora",
    "bruguiera_parviflora",
    "bruguiera parviflora",
    "ceriops-tagal",
    "ceriops_tagal",
    "ceriops tagal",
    "acanthus-ilicifolius",
    "acanthus_ilicifolius",
    "acanthus ilicifolius",
    "acrostichum-aureum",
    "acrostichum_aureum",
    "acrostichum aureum",
    "nipah-fruticans",
    "nipah_fruticans",
    "nipah fruticans",
)

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
    """Species-only (13-class) classifier: ALWAYS treat the top argmax as the answer.

    User requested to REMOVE out-of-distribution handling so the UI behaves like the first
    version: never "Not part of 13 trained mangrove species", never show unknown banners,
    and always just display the single best-guess species from the 13 trained labels,
    together with its raw confidence meter.

    We still compute diagnostic numbers (entropy / margin etc.) inside the returned dict
    because they are harmless for future debugging, but `is_unknown` is forced False.
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
    return {
        "is_unknown": False,
        "reasons": [],
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
        "roboflow": {
            "configured": bool(ROBOFLOW_API_KEY and ROBOFLOW_HEALTH_MODEL),
            "model": ROBOFLOW_HEALTH_MODEL or None,
            "base_url": ROBOFLOW_BASE_URL,
            "timeout_s": ROBOFLOW_TIMEOUT_S,
        },
    }


# ----------------------------------------------------------------------
# Roboflow helper functions — keep the response contract JSON-safe.
# ----------------------------------------------------------------------
def _image_to_bytes_jpeg(img: Image.Image, max_edge: int = 1280, quality: int = 85) -> bytes:
    """Resize preserving aspect, JPEG compress, return raw bytes."""
    w, h = img.size
    scale = min(1.0, float(max_edge) / float(max(w, h)))
    if scale < 1.0:
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def _image_to_roboflow_base64(img: Image.Image, max_edge: int = 1280, quality: int = 85) -> str:
    """Resize to max_edge preserving aspect, JPEG compress, then return data URI.

    Roboflow's Hosted API (detect.roboflow.com) accepts:
      1. multipart file upload
      2. data URI base64 inside JSON body (smaller POSTs for mobile).
    """
    raw = _image_to_bytes_jpeg(img, max_edge=max_edge, quality=quality)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_roboflow_class(raw: str) -> str:
    return (raw or "").strip().replace("_", " ")


def _severity_for_class(name: str) -> str | None:
    """Return mild / moderate / severe based on keywords in class names.

    If your Roboflow classes are named like:
      - Healthy / Normal / No Disease → severity = None (healthy case)
      - Leaf Spot Mild / Anthracnose Mild → mild
      - Leaf Spot Moderate → moderate
      - Severe Blight / Dieback Severe → severe
    we map automatically. Otherwise, severity = None and UI leaves it off.
    """
    n = name.lower()
    if any(k in n for k in ("severe", "critical", "advanced", "heavy")):
        return "severe"
    if any(k in n for k in ("moderate", "mid", "medium")):
        return "moderate"
    if any(k in n for k in ("mild", "early", "light", "slight", "minor", "low")):
        return "mild"
    return None


def _roboflow_is_healthy(class_name: str) -> bool:
    """Return True when the Roboflow class is explicitly "healthy" or a neutral species/propagule class.

    Universe project "Mangrove_Species_Diseases" has mixed classes:
      - Stress classes: black_spots, brown_spots, dieback-gall  => _is_healthy = False
      - Neutral classes (mangrove species / flowers / propagules detected on the leaf)
          => treated as "not stressed" => _is_healthy = True
      - Explicit Healthy/Normal classes: True
      - Anything unrecognized => False (flag for LGU review)
    """
    n = (class_name or "").strip().lower()
    if not n:
        return False
    if n in {c.lower() for c in ROBOFLOW_DISEASE_CLASSES_EXACT}:
        return False
    if n in {c.lower() for c in ROBOFLOW_NEUTRAL_CLASSES_EXACT}:
        return True
    if any(k in n for k in ROBOFLOW_DISEASE_CLASSES_EXACT):
        # Ex: "black_spots_early", "dieback-gall_severe" — still disease
        return False
    if any(k in n for k in ROBOFLOW_HEALTHY_KEYWORDS):
        return True
    if any(k in n for k in ROBOFLOW_STRESS_KEYWORDS):
        return False
    # Undetermined: treat as unhealthy so UI flags for LGU review.
    return False


def _roboflow_is_serverless(base_url: str) -> bool:
    """Decide Roboflow upload style based on env base URL.

    serverless.roboflow.com uses multipart file POST (consistent with inference_sdk InferenceHTTPClient.infer).
    detect.roboflow.com   uses data-uri JSON body upload with api_key query param.
    """
    return "serverless.roboflow.com" in (base_url or "").lower()


def _normalize_roboflow_response(payload: dict) -> dict:
    """Take Roboflow JSON (Hosted API or Serverless API) and return Mangrow-standard shape.

    Normalizes all of these inputs into one stable shape:
      1. Object-Detection project:
          {"predictions": [ {"class": "...", "confidence": 0.xx, "x":.., "y":.., "width":.., "height":..} ] }
      2. Classification project (Hosted API):
          {"predicted_classes": [...], "predictions": { "<class>": <conf>, ... }, "top": "<class>", "confidence": 0.xx }
      3. Classification project (Serverless API / inference_sdk):
          {"predictions": [ {"class": "<class>", "class_id":..., "confidence": 0.xx}, ... ], ... }
    """
    if not isinstance(payload, dict):
        raise HTTPException(502, "Roboflow returned a non-JSON response.")

    # ----- Variant A: Hosted Classification: predictions is a MAP of {class: conf} -----
    if isinstance(payload.get("predictions"), dict) and (payload.get("top") or payload.get("predicted_classes")):
        raw_cls_map = payload["predictions"]
        items = sorted(
            ({"class": str(cls), "confidence": float(conf)} for cls, conf in raw_cls_map.items()),
            key=lambda r: -r["confidence"],
        )[:5]
    elif isinstance(payload.get("predictions"), list):
        # ----- Variant B / C: list of predictions (either Object-Detection bboxes, or Serverless Classification list) -----
        raw_list: list = payload["predictions"]
        is_classification_list = bool(raw_list) and all(
            isinstance(x, dict) and "class" in x and "confidence" in x and "x" not in x
            for x in raw_list[:3]
        )
        if is_classification_list:
            # Serverless-style classification list: [{class, class_id, confidence}]
            best_by_cls: dict[str, dict] = {}
            for d in raw_list:
                if not isinstance(d, dict):
                    continue
                cls_raw = _parse_roboflow_class(str(d.get("class", "")))
                if not cls_raw:
                    continue
                conf = float(d.get("confidence") or 0.0)
                prev = best_by_cls.get(cls_raw)
                if prev is None or conf > float(prev["confidence"]):
                    best_by_cls[cls_raw] = {"class": cls_raw, "confidence": conf}
            items = sorted(best_by_cls.values(), key=lambda r: -float(r["confidence"]))[:5]
        else:
            # Object-Detection list: dedupe per class keeping max conf, keep bbox.
            best_by_cls = {}
            for d in raw_list:
                if not isinstance(d, dict):
                    continue
                cls_raw = _parse_roboflow_class(str(d.get("class", "")))
                if not cls_raw:
                    continue
                conf = float(d.get("confidence") or 0.0)
                prev = best_by_cls.get(cls_raw)
                if prev is None or conf > float(prev["confidence"]):
                    best_by_cls[cls_raw] = {
                        "class": cls_raw,
                        "confidence": conf,
                        "bbox": {
                            "x": float(d.get("x") or 0.0),
                            "y": float(d.get("y") or 0.0),
                            "width": float(d.get("width") or 0.0),
                            "height": float(d.get("height") or 0.0),
                        }
                        if all(k in d for k in ("x", "y", "width", "height"))
                        else None,
                    }
            items = sorted(best_by_cls.values(), key=lambda r: -float(r["confidence"]))[:5]
    else:
        items = []

    if not items:
        # No detections / no classes → UI shows uncertain.
        predictions = []
        top_health = None
        is_healthy = False
        uncertain = True
    else:
        top = items[0]
        top_name = top["class"]
        top_conf = float(top["confidence"])
        is_healthy = _roboflow_is_healthy(top_name)
        uncertain = bool(top_conf < ROBOFLOW_UNCERTAIN_CONF)
        predictions = [
            {
                "class_name": it["class"],
                "display_name": it["class"].replace("_", " ").replace("-", " ").title(),
                "confidence": float(it["confidence"]),
                "severity": _severity_for_class(it["class"]),
                "is_healthy": _roboflow_is_healthy(it["class"]),
                **({"bbox": it["bbox"]} if it.get("bbox") is not None else {}),
            }
            for it in items
        ]
        top_health = {
            "status": "healthy" if is_healthy else "unhealthy",
            "label": "Normal" if is_healthy else "Shows Stress",
            "description": None,
            "driving_class": predictions[0]["display_name"],
            "confidence": predictions[0]["confidence"],
        }

    return {
        "predictions": predictions,
        "uncertain": uncertain,
        "top_health": top_health,
        "raw": payload,  # Pass back raw for debugging.
    }


@app.post("/health-predict")
async def health_predict(image: UploadFile = File(...)):
    """Proxy the uploaded leaf image to Roboflow (Hosted or Serverless) and return Mangrow-standard JSON.

    Security: the Roboflow secret API key is never exposed to the Expo app.
    This endpoint accepts the same multipart `image` field as /predict.

    Upload strategy (auto-detected based on ROBOFLOW_BASE_URL):
      - serverless.roboflow.com -> multipart file POST (like inference_sdk InferenceHTTPClient.infer)
      - detect.roboflow.com     -> data-uri JSON body (smaller POST, avoids 413 for phone photos)
    """
    if not ROBOFLOW_API_KEY or not ROBOFLOW_HEALTH_MODEL:
        raise HTTPException(
            503,
            "Roboflow health AI is not configured on this server. "
            "Set ROBOFLOW_API_KEY + ROBOFLOW_HEALTH_MODEL environment variables.",
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Send an image multipart field named 'image'")

    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        img = open_image_rgb(data)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    use_serverless = _roboflow_is_serverless(ROBOFLOW_BASE_URL)

    if use_serverless:
        # Match inference_sdk style: multipart file POST to serverless.roboflow.com/<model_id>?api_key=...
        jpeg_bytes = _image_to_bytes_jpeg(img, max_edge=1280, quality=85)
        url = f"{ROBOFLOW_BASE_URL}/{ROBOFLOW_HEALTH_MODEL}"
        params = {"api_key": ROBOFLOW_API_KEY}
        files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
        try:
            async with httpx.AsyncClient(timeout=ROBOFLOW_TIMEOUT_S) as client:
                resp = await client.post(url, params=params, files=files)
        except httpx.TimeoutException as e:
            raise HTTPException(504, f"Roboflow request timed out after {ROBOFLOW_TIMEOUT_S}s.") from e
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Roboflow network error: {e}") from e
    else:
        # detect.roboflow.com -> data-uri JSON body with api_key query param
        try:
            data_uri = _image_to_roboflow_base64(img, max_edge=1280, quality=85)
        except Exception as e:
            raise HTTPException(500, f"Failed to encode image for Roboflow: {e}") from e
        url = f"{ROBOFLOW_BASE_URL}/{ROBOFLOW_HEALTH_MODEL}"
        params = {"api_key": ROBOFLOW_API_KEY}
        body = {"image": data_uri}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=ROBOFLOW_TIMEOUT_S) as client:
                resp = await client.post(url, params=params, json=body, headers=headers)
        except httpx.TimeoutException as e:
            raise HTTPException(504, f"Roboflow request timed out after {ROBOFLOW_TIMEOUT_S}s.") from e
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Roboflow network error: {e}") from e

    try:
        payload = resp.json()
    except Exception as e:
        raise HTTPException(502, f"Roboflow returned invalid JSON (HTTP {resp.status}).") from e

    if resp.status_code >= 400:
        raise HTTPException(
            resp.status_code,
            f"Roboflow error: {payload.get('message') or payload.get('error') or f'HTTP {resp.status}'}",
        )

    normalized = _normalize_roboflow_response(payload)
    return normalized


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
