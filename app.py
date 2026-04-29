import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from scipy.interpolate import Rbf


@dataclass
class ReconstructionConfig:
    depth_scale: float = 140.0
    smooth_sigma: float = 2.2
    outlier_clip: float = 2.5


mp_face_mesh = mp.solutions.face_mesh


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _detect_landmarks(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        result = mesh.process(image)

    if not result.multi_face_landmarks:
        raise ValueError("No face detected. Use a clearer, frontal image with better lighting.")

    points = []
    for lm in result.multi_face_landmarks[0].landmark:
        points.append((lm.x * w, lm.y * h))
    return np.array(points, dtype=np.float32)


def _estimate_depth_from_profile(front_pts: np.ndarray, profile_pts: np.ndarray) -> np.ndarray:
    # Approximate per-point depth using x-axis displacement between frontal and profile landmarks.
    # This is a heuristic and should not be used as forensic evidence.
    delta = np.abs(front_pts[:, 0] - profile_pts[:, 0])
    z = (delta - np.median(delta))
    mad = np.median(np.abs(z - np.median(z))) + 1e-6
    z = np.clip(z / (1.4826 * mad), -3, 3)
    return z


def _fit_dense_depth(pts_xy: np.ndarray, pts_z: np.ndarray, shape: Tuple[int, int], cfg: ReconstructionConfig) -> np.ndarray:
    h, w = shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    rbf = Rbf(pts_xy[:, 0], pts_xy[:, 1], pts_z, function="multiquadric", smooth=cfg.smooth_sigma)
    dense_z = rbf(gx, gy)

    z_mean, z_std = dense_z.mean(), dense_z.std() + 1e-6
    dense_z = np.clip((dense_z - z_mean) / z_std, -cfg.outlier_clip, cfg.outlier_clip)
    dense_z *= cfg.depth_scale
    return dense_z.astype(np.float32)


def _render_depth_preview(depth: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _write_point_cloud(rgb: np.ndarray, depth: np.ndarray, stride: int = 3) -> str:
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]

    x = xs.astype(np.float32) - (w / 2)
    y = ys.astype(np.float32) - (h / 2)
    z = depth[0:h:stride, 0:w:stride]

    colors = rgb[0:h:stride, 0:w:stride].reshape(-1, 3)
    points = np.stack([x.reshape(-1), -y.reshape(-1), z.reshape(-1)], axis=1)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    ply_path = tmp.name
    tmp.close()

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (px, py, pz), (r, g, b) in zip(points, colors):
            f.write(f"{px:.3f} {py:.3f} {pz:.3f} {int(r)} {int(g)} {int(b)}\n")
    return ply_path


def reconstruct(front_image_path: str, profile_image_path: str, depth_scale: float, smoothing: float) -> Tuple[np.ndarray, str, str]:
    cfg = ReconstructionConfig(depth_scale=depth_scale, smooth_sigma=smoothing)
    front = _read_image(front_image_path)
    profile = _read_image(profile_image_path)

    front_pts = _detect_landmarks(front)
    profile_pts = _detect_landmarks(profile)

    z_sparse = _estimate_depth_from_profile(front_pts, profile_pts)
    depth = _fit_dense_depth(front_pts, z_sparse, front.shape[:2], cfg)

    preview = _render_depth_preview(depth)
    ply_path = _write_point_cloud(front, depth)

    report = (
        "Reconstruction completed. Output is an investigative aid only and is NOT courtroom-grade evidence. "
        "Validate with controlled photogrammetry and professional forensic workflows."
    )
    return preview, ply_path, report


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CSI Facial Reconstruction (Python ML Prototype)") as demo:
        gr.Markdown(
            "# CSI Facial Reconstruction (Prototype)\n"
            "Upload one frontal and one profile face image to generate a depth map and 3D point cloud.\n"
            "**For training/research only. Not suitable for legal identification.**"
        )
        with gr.Row():
            front = gr.Image(type="filepath", label="Frontal Image")
            profile = gr.Image(type="filepath", label="Profile Image")

        with gr.Row():
            depth_scale = gr.Slider(60, 240, value=140, step=5, label="Depth scale")
            smoothing = gr.Slider(0.1, 5.0, value=2.2, step=0.1, label="RBF smoothing")

        run_btn = gr.Button("Reconstruct Face")

        depth_preview = gr.Image(label="Depth Preview", type="numpy")
        point_cloud = gr.File(label="Download 3D point cloud (.ply)")
        report = gr.Textbox(label="Status", lines=3)

        run_btn.click(
            fn=reconstruct,
            inputs=[front, profile, depth_scale, smoothing],
            outputs=[depth_preview, point_cloud, report],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
