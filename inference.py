import base64
import io
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

CLASSIFICATION_CKPT = "deploy_effb3.pth"
SEGMENTATION_CKPT = "models/segmentation/best_unet_model.pth"

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


class TemperatureScaler(nn.Module):
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature


def _tta_transforms(image_size: int):
    mean, std = list(IMG_MEAN), list(IMG_STD)
    return [
        A.Compose([A.Resize(image_size, image_size), A.Normalize(mean, std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1), A.Normalize(mean, std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=10, p=1), A.Normalize(mean, std), ToTensorV2()]),
    ]


class BrainTumorClassifier:
    def __init__(self, ckpt_path: str):
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Model not found at '{ckpt_path}'")

        bundle = torch.load(ckpt_path, map_location="cpu")
        cfg = bundle["config"]

        self.class_names = bundle["class_names"]
        self.threshold = float(bundle["threshold"])
        self.image_size = int(cfg["image_size"])
        self.tta_n = int(cfg.get("tta_n", 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base = timm.create_model(
            cfg["model_name"],
            pretrained=False,
            num_classes=cfg["num_classes"],
            drop_rate=cfg["drop_rate"],
        )
        base.load_state_dict(bundle["model_state"])

        self.model = TemperatureScaler(base, float(bundle.get("temperature", 1.0)))
        self.model.to(self.device).eval()
        self._tfs = _tta_transforms(self.image_size)[: self.tta_n]

        print(f"[classifier] device={self.device} classes={self.class_names} threshold={self.threshold}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        img_np = np.array(image.convert("RGB"))
        probs = np.zeros(len(self.class_names), dtype=np.float32)

        for tf in self._tfs:
            tensor = tf(image=img_np)["image"].unsqueeze(0).to(self.device)
            out = self.model(tensor)
            probs += F.softmax(out, dim=1).squeeze().cpu().numpy()

        probs /= len(self._tfs)
        pred = int(probs.argmax())
        confidence = float(probs[pred])
        uncertain = confidence < self.threshold

        warning = None
        if uncertain:
            warning = f"Low confidence ({confidence*100:.1f}%). Please consult a specialist."

        return {
            "predicted_class": self.class_names[pred],
            "confidence": round(confidence, 4),
            "predictions": {cls: round(float(p), 4) for cls, p in zip(self.class_names, probs)},
            "uncertainity": bool(uncertain),
            "warning": warning,
        }


class BrainTumorSegmenter:
    def __init__(self, ckpt_path: str, device: str | None = None):
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found at '{ckpt_path}'")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.encoder = ckpt["encoder"]
        self.img_size = int(ckpt["img_size"])

        self.model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=None,
            in_channels=int(ckpt["in_channels"]),
            classes=int(ckpt["num_classes"]),
            activation=None,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.val_dice = float(ckpt.get("val_dice", 0.0))
        self.timestamp = str(ckpt.get("timestamp", "N/A"))

        print(f"[segmenter] device={self.device} encoder={self.encoder} img_size={self.img_size}")

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        resized = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        mean = np.array(IMG_MEAN, dtype=np.float32)
        std = np.array(IMG_STD, dtype=np.float32)
        img = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, resized

    @torch.no_grad()
    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        rgb = np.array(image.convert("RGB"))
        tensor, resized = self._preprocess(rgb)

        logit = self.model(tensor)  # (1,1,H,W)
        prob = torch.sigmoid(logit).squeeze().float().cpu().numpy()  # (H,W) in [0,1]
        mask = (prob > float(threshold)).astype(np.uint8) * 255

        tumor_px = int((mask > 0).sum())
        total_px = int(mask.size)
        coverage = float(100.0 * tumor_px / max(1, total_px))

        overlay = resized.copy()
        alpha = 0.45
        color = np.array([0, 210, 255], dtype=np.float32)  # RGB-ish tone, but we are in RGB already
        bool_mask = mask > 0
        overlay[bool_mask] = np.clip(
            overlay[bool_mask].astype(np.float32) * (1 - alpha) + color * alpha, 0, 255
        ).astype(np.uint8)

        prob_u8 = (prob * 255).astype(np.uint8)
        prob_color = cv2.applyColorMap(prob_u8, cv2.COLORMAP_PLASMA)
        prob_color = cv2.cvtColor(prob_color, cv2.COLOR_BGR2RGB)

        return {
            "mask": mask,
            "prob": prob,
            "prob_color": prob_color,
            "overlay": overlay,
            "resized": resized,
            "coverage": round(coverage, 3),
            "tumor_px": tumor_px,
            "total_px": total_px,
        }


def _validate_image_upload(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG/PNG allowed")


def _png_base64(rgb_or_gray: np.ndarray) -> str:
    if rgb_or_gray.ndim == 2:
        arr = rgb_or_gray
    else:
        # cv2 expects BGR for encoding; convert RGB->BGR
        arr = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

classifier: BrainTumorClassifier | None = None
segmenter: BrainTumorSegmenter | None = None


@app.on_event("startup")
def _load_models():
    global classifier, segmenter
    classifier = BrainTumorClassifier(CLASSIFICATION_CKPT)
    segmenter = BrainTumorSegmenter(SEGMENTATION_CKPT)


app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")


@app.get("/")
def root():
    return FileResponse("static/index.html", media_type="text/html")


@app.get("/favicon.svg")
def favicon():
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")


@app.get("/icons.svg")
def icons():
    return FileResponse("static/icons.svg", media_type="image/svg+xml")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_device": str(classifier.device) if classifier else None,
        "segmenter_device": str(segmenter.device) if segmenter else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")

    _validate_image_upload(file)
    t0 = time.time()
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = classifier.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result["inference_time"] = round((time.time() - t0) * 1000, 1)
    result["filename"] = file.filename
    return result


@app.post("/segment")
async def segment(file: UploadFile = File(...), threshold: float = 0.5):
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Segmenter not loaded")

    _validate_image_upload(file)
    t0 = time.time()
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        out = segmenter.predict(image, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename": file.filename,
        "threshold": float(threshold),
        "coverage": out["coverage"],
        "tumor_px": out["tumor_px"],
        "total_px": out["total_px"],
        "img_size": segmenter.img_size,
        "encoder": segmenter.encoder,
        "val_dice": round(segmenter.val_dice, 6),
        "timestamp": segmenter.timestamp,
        "inference_time": round((time.time() - t0) * 1000, 1),
        "mask_png": _png_base64(out["mask"]),
        "overlay_png": _png_base64(out["overlay"]),
        "prob_png": _png_base64(out["prob_color"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference:app", host="127.0.0.1", port=8000, reload=False)