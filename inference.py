"""
Inference utilities for the building roof segmentation demo.

Loads the fine-tuned Mask R-CNN (ResNet-50 FPN) and draws predictions in the
same style as the original project notebook:
  - ground truth: green masks with red boxes
  - predictions:  blue (low confidence) -> purple (high confidence), red boxes
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Colours match the project notebook (RGB)
GT_MASK_COLOR = np.array([0, 255, 0])
PRED_COLOR_LOW = np.array([0, 0, 255])
PRED_COLOR_HIGH = np.array([138, 43, 226])
BOX_COLOR = (255, 0, 0)

# Weights are NOT stored in this repo. Point MODEL_PATH at your local file,
# or set HF_MODEL_REPO to load them from a private Hugging Face model repo.
DEFAULT_MODEL_PATH = "weights/optimised_config.pt"


def _resolve_weights() -> str:
    local = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    if Path(local).exists():
        return local
    repo = os.environ.get("HF_MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=repo,
            filename=os.environ.get("HF_MODEL_FILE", "optimised_config.pt"),
            token=os.environ.get("HF_TOKEN"),
        )
    raise FileNotFoundError(
        f"Model weights not found at '{local}'. Set MODEL_PATH to your weights file."
    )


def load_model(device: str = "cpu") -> torch.nn.Module:
    """Build Mask R-CNN with a 2-class head (background + building) and load weights."""
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    state = torch.load(_resolve_weights(), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_rgb(path_or_array) -> np.ndarray:
    """Return an HxWx3 uint8 RGB array from a file path or array."""
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
    else:
        img = np.array(Image.open(path_or_array))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[:, :, :3].astype(np.uint8)


@torch.no_grad()
def predict(model, img: np.ndarray, score_threshold: float = 0.5, device: str = "cpu") -> dict:
    """Run the model on one RGB image and keep detections above the threshold."""
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
    out = model([tensor])[0]
    keep = out["scores"] >= score_threshold
    return {k: out[k][keep].cpu() for k in ("boxes", "scores", "masks")}


def draw_predictions(img: np.ndarray, pred: dict, score_threshold: float = 0.5,
                     alpha: float = 0.7) -> np.ndarray:
    overlay = img.copy()
    for mask, score in zip(pred["masks"], pred["scores"].numpy()):
        t = np.clip((score - score_threshold) / max(1e-6, 1.0 - score_threshold), 0, 1)
        color = (PRED_COLOR_LOW * (1 - t) + PRED_COLOR_HIGH * t).astype(np.uint8)
        m = (mask.squeeze() > 0.5).numpy()
        overlay[m] = (overlay[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    for box in pred["boxes"].numpy().astype(int):
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)
    return overlay


class GroundTruth:
    """Looks up COCO-format labels for the bundled sample tiles."""

    def __init__(self, json_path: str):
        data = json.load(open(json_path))
        by_id = {im["id"]: im["file_name"] for im in data["images"]}
        self.anns = {}
        for a in data["annotations"]:
            name = by_id[a["image_id"]].replace("tile_", "uc_")
            self.anns.setdefault(name, []).append(a)

    def has(self, filename: str) -> bool:
        return Path(filename).name in self.anns

    def count(self, filename: str) -> int:
        return len(self.anns.get(Path(filename).name, []))

    def draw(self, img: np.ndarray, filename: str, alpha: float = 0.7) -> np.ndarray:
        overlay = img.copy()
        anns = self.anns.get(Path(filename).name, [])
        for a in anns:
            m = mask_utils.decode(a["segmentation"]).astype(bool)
            overlay[m] = (overlay[m] * (1 - alpha) + GT_MASK_COLOR * alpha).astype(np.uint8)
        for a in anns:
            x, y, w, h = [int(v) for v in a["bbox"]]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), BOX_COLOR, 2)
        return overlay
