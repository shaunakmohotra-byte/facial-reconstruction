# CSI Facial Reconstruction (Python ML Prototype)

A lightweight Python app for **investigative prototyping** of facial reconstruction from two images (frontal + profile). It produces:

- A pseudo-depth map preview.
- A downloadable 3D point cloud (`.ply`).

> ⚠️ This project is **not** forensic-grade biometric identification software and must not be used as courtroom evidence.

## Approach

1. Detect face landmarks with MediaPipe Face Mesh.
2. Estimate sparse depth heuristically from frontal/profile landmark displacement.
3. Interpolate dense depth with radial basis functions (RBF).
4. Export a colorized point cloud.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

## Notes for CSI-style workflows

- Use high-resolution, well-lit photos with neutral expression.
- Prefer camera metadata and known focal lengths if available.
- Treat output as a lead-generation aid only.
