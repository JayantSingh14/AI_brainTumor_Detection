import io
import time
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore")

DEPLOY_PATH = "deploy_effb3.pth"


class TemperatureScaler(nn.Module):
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, x):
        return self.model(x) / self.temperature


def tta_transforms(image_size: int):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [
        A.Compose([A.Resize(image_size, image_size), A.Normalize(mean, std), ToTensorV2()]),
        A.Compose(
            [A.Resize(image_size, image_size), A.HorizontalFlip(p=1), A.Normalize(mean, std), ToTensorV2()]
        ),
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=10, p=1), A.Normalize(mean, std), ToTensorV2()]),
    ]


class BrainTumorClassifier:
    def __init__(self, deploy_path: str):
        if not Path(deploy_path).exists():
            raise FileNotFoundError(f"Model not found at '{deploy_path}'")

        bundle = torch.load(deploy_path, map_location="cpu")
        cfg = bundle["config"]

        self.class_names = bundle["class_names"]
        self.threshold = bundle["threshold"]
        self.image_size = cfg["image_size"]
        self.tta_n = cfg.get("tta_n", 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base = timm.create_model(
            cfg["model_name"],
            pretrained=False,
            num_classes=cfg["num_classes"],
            drop_rate=cfg["drop_rate"],
        )
        base.load_state_dict(bundle["model_state"])

        self.model = TemperatureScaler(base, bundle.get("temperature", 1.0))
        self.model.to(self.device).eval()

        self._tfs = tta_transforms(self.image_size)[: self.tta_n]

        print(f"Loaded model on {self.device}")
        print(f"Classes: {self.class_names}")
        print(f"Threshold: {self.threshold}")

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
            "uncertainity": uncertain,
            "warning": warning,
        }


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

classifier: BrainTumorClassifier | None = None


@app.on_event("startup")
def _load():
    global classifier
    classifier = BrainTumorClassifier(DEPLOY_PATH)


# serve built UI
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
    return {"status": "ok", "device": str(classifier.device) if classifier else None}


def _validate(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG/PNG allowed")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _validate(file)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference:app", host="127.0.0.1", port=8000, reload=False)

import io
import time
import warnings
from pathlib import Path

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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore")

DEPLOY_PATH = "deploy_effb3.pth"


class TemperatureScaler(nn.Module):
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, x):
        return self.model(x) / self.temperature


def tta_transforms(image_size: int):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [
        A.Compose([A.Resize(image_size, image_size),
                   A.Normalize(mean, std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1),
                   A.Normalize(mean, std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=10, p=1),
                   A.Normalize(mean, std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size),
                   A.RandomBrightnessContrast(0.12, 0.12, p=1),
                   A.Normalize(mean, std), ToTensorV2()]),
        A.Compose([A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                   A.CenterCrop(image_size, image_size),
                   A.Normalize(mean, std), ToTensorV2()]),
    ]


class BrainTumorClassifier:
    def __init__(self, deploy_path: str):
        if not Path(deploy_path).exists():
            raise FileNotFoundError(f"Model not found at '{deploy_path}'")

        bundle = torch.load(deploy_path, map_location="cpu")

        cfg = bundle["config"]
        self.class_names = bundle["class_names"]
        self.threshold = bundle["threshold"]
        self.image_size = cfg["image_size"]
        self.tta_n = cfg["tta_n"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base = timm.create_model(
            cfg["model_name"],
            pretrained=False,
            num_classes=cfg["num_classes"],
            drop_rate=cfg["drop_rate"],
        )
        base.load_state_dict(bundle["model_state"])

        self.model = TemperatureScaler(base, bundle["temperature"])
        self.model.to(self.device).eval()

        self._tfs = tta_transforms(self.image_size)[: self.tta_n]

        print(f"Loaded model on {self.device}")
        print(f"Classes: {self.class_names}")
        print(f"Threshold: {self.threshold}")

    @torch.no_grad()
    def predict(self, image_input) -> dict:
        img_np = self._load_image(image_input)

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
        elif self.class_names[pred] == "meningioma" and confidence < 0.85:
            warning = "Meningioma predicted with moderate confidence. Consider review."

        return {
            "predicted_class": self.class_names[pred],
            "confidence": round(confidence, 4),
            "predictions": {
                cls: round(float(p), 4)
                for cls, p in zip(self.class_names, probs)
            },
            "uncertainity": uncertain,
            "warning": warning,
        }

    @staticmethod
    def _load_image(image_input) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            return np.array(Image.open(image_input).convert("RGB"))
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

classifier: BrainTumorClassifier = None


@app.on_event("startup")
def load_model():
    global classifier
    classifier = BrainTumorClassifier(DEPLOY_PATH)


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
        "device": str(classifier.device) if classifier else None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _validate_image_file(file)

    t0 = time.time()
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = classifier.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result["inference_time"] = round((time.time() - t0) * 1000, 1)
    result["filename"] = file.filename

    return JSONResponse(content=result)


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images allowed")

    results = []
    for f in files:
        t0 = time.time()
        try:
            _validate_image_file(f)
            img_bytes = await f.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            result = classifier.predict(image)
            result["filename"] = f.filename
            result["inference_time"] = round((time.time() - t0) * 1000, 1)
        except HTTPException as e:
            result = {"filename": f.filename, "error": e.detail}
        except Exception as e:
            result = {"filename": f.filename, "error": str(e)}

        results.append(result)

    return JSONResponse(content={"count": len(results), "predictions": results})


def _validate_image_file(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG/PNG allowed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host="127.0.0.1", port=8000, reload=False)