# Brain Tumor MRI Classifier (Desktop UI)

- **Frontend**: React + Tailwind + Recharts + Framer Motion (desktop-only layout)
- **Backend**: FastAPI + PyTorch inference

## Run

```bash
chmod +x run.sh
./run.sh
```

Open:
- App: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Dev (frontend)

```bash
cd frontend
npm run dev
```

The frontend calls the API at `http://localhost:8000/predict`.

## GitHub notes

- The model weights file `deploy_effb3.pth` is **ignored by default** in `.gitignore`.
- If you want to store model weights in GitHub, use **Git LFS** (recommended) or upload it as a **Release asset**.

