#!/usr/bin/env python3
"""
AI-Based Waste Classification System - Gradio Web Application

Features:
- Image Classification: upload JPG/JPEG/PNG and get predicted class, confidence, and probabilities
- Video Classification: upload a video, sample up to 20 frames evenly, classify whole frames with the
  same trained CNN, and display averaged predictions (whole-frame classification, not detection/tracking)

Notes:
- Uses the existing trained model (models/best_waste_cnn.keras) and class names (models/class_names.json)
- Model is loaded once at startup (CPU-compatible)
- Does NOT retrain or modify the model/dataset/training artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

try:
    import cv2  # OpenCV for video frame extraction
    CV2_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on environment
    cv2 = None
    CV2_IMPORT_ERROR = str(e)

# -----------------------------
# Configuration (must match training preprocessing)
# -----------------------------
MODEL_PATH = Path("models/best_waste_cnn.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
IMG_SIZE = (96, 96)  # same resize target used during training
DISPLAY_CLASS_NAMES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
MAX_VIDEO_FRAMES = 20  # sample at most 20 frames evenly to keep CPU usage low


def load_model_and_classes():
    """Load the saved trained model and class names (read-only)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_names = data.get("class_names")
    if not isinstance(class_names, list) or len(class_names) == 0:
        raise ValueError("Invalid class_names.json: expected a non-empty 'class_names' list")

    # Basic compatibility check (6 classes expected for this project)
    if len(class_names) != 6:
        print(f"WARNING: Expected 6 classes, found {len(class_names)}: {class_names}")

    return model, class_names


# Load once at startup (CPU inference, lightweight)
try:
    MODEL, CLASS_NAMES = load_model_and_classes()
    LOAD_ERROR = None
except Exception as e:
    MODEL, CLASS_NAMES = None, []
    LOAD_ERROR = str(e)


def format_class_label(name: str) -> str:
    """Format class labels for display (e.g., 'cardboard' -> 'Cardboard')."""
    return name.replace("_", " ").strip().title()


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Apply the same preprocessing/normalization used during CNN training:
    1) Convert to RGB
    2) Resize to 96x96
    3) Convert to float32 and normalize pixel values to [0, 1]
    4) Add batch dimension -> shape (1, 96, 96, 3)
    """
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def make_probability_dataframe(class_names: list[str], probs: np.ndarray) -> pd.DataFrame:
    """Create a DataFrame for charting probabilities across all classes."""
    labels = [format_class_label(c) for c in class_names]
    values = [float(p) for p in probs]

    # Keep a consistent display order if standard classes are present
    lower_map = {c.lower(): i for i, c in enumerate(class_names)}
    if all(name.lower() in lower_map for name in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]):
        ordered = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        labels = [format_class_label(c) for c in ordered]
        values = [float(probs[lower_map[c]]) for c in ordered]

    df = pd.DataFrame({"Category": labels, "Probability": values})
    return df


def empty_probability_outputs():
    """Return empty/default probability chart and label outputs."""
    df = pd.DataFrame({"Category": DISPLAY_CLASS_NAMES, "Probability": [0, 0, 0, 0, 0, 0]})
    return df, {row["Category"]: 0.0 for _, row in df.iterrows()}


# -----------------------------
# Image classification (preserved)
# -----------------------------

def predict_waste(image: Image.Image | None):
    """Run image inference and return UI outputs (preview, status, confidence, chart, labels)."""
    empty_df, empty_labels = empty_probability_outputs()

    if LOAD_ERROR:
        msg = (
            f"### Unable to load model/resources\n"
            f"`{LOAD_ERROR}`\n\n"
            f"Please ensure `models/best_waste_cnn.keras` and `models/class_names.json` exist."
        )
        return image, msg, "", empty_df, empty_labels

    if image is None:
        return None, "### Please upload a JPG, JPEG, or PNG image to begin.", "", empty_df, empty_labels

    if MODEL is None or not CLASS_NAMES:
        return image, "### Model is not available.", "", empty_df, empty_labels

    # Preprocess + predict (reused preprocessing)
    x = preprocess_image(image)
    probs = MODEL.predict(x, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float32)

    # Predicted class
    top_idx = int(np.argmax(probs))
    pred_name = format_class_label(CLASS_NAMES[top_idx])
    confidence = float(probs[top_idx])

    # Probability outputs (all categories)
    prob_df = make_probability_dataframe(CLASS_NAMES, probs)
    prob_dict = {row["Category"]: float(row["Probability"]) for _, row in prob_df.iterrows()}

    prediction_md = (
        f"## Predicted Category: **{pred_name}**\n"
        f"**Confidence:** {confidence:.2%}"
    )
    confidence_md = f"Model confidence in the top prediction: **{confidence:.2%}**"

    return image, prediction_md, confidence_md, prob_df, prob_dict


def clear_image_all():
    """Reset all image classification UI components."""
    empty_df, empty_labels = empty_probability_outputs()
    return None, None, "### Upload an image to see the prediction.", "", empty_df, empty_labels


# -----------------------------
# Video classification (new)
# -----------------------------

def get_video_duration_seconds(frame_count: int, fps: float) -> float | None:
    """Compute video duration from frame count and FPS when available."""
    if fps and fps > 0 and frame_count > 0:
        return float(frame_count / fps)
    return None


def sample_frame_indices(frame_count: int, max_frames: int = MAX_VIDEO_FRAMES) -> np.ndarray:
    """Select up to max_frames indices evenly throughout the video."""
    if frame_count <= 0:
        return np.array([], dtype=int)
    n = min(max_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, num=n, dtype=int)
    # Ensure unique, sorted indices (linspace can duplicate for short videos)
    return np.unique(indices)


def predict_waste_video(video_path: str | None):
    """Classify a video by sampling whole frames and averaging CNN probabilities."""
    empty_df, empty_labels = empty_probability_outputs()

    if LOAD_ERROR:
        msg = (
            f"### Unable to load model/resources\n"
            f"`{LOAD_ERROR}`\n\n"
            f"Please ensure `models/best_waste_cnn.keras` and `models/class_names.json` exist."
        )
        return video_path, msg, "", empty_df, empty_labels, "Frames analysed: 0"

    if cv2 is None:
        msg = (
            "### OpenCV (cv2) is not available\n"
            "Video classification requires OpenCV. Install it in your environment and restart the app.\n\n"
            f"**Import error:** `{CV2_IMPORT_ERROR}`"
        )
        return video_path, msg, "", empty_df, empty_labels, "Frames analysed: 0"

    if video_path is None or (isinstance(video_path, str) and not video_path.strip()):
        return None, "### Please upload a video (MP4, AVI, etc.) to begin.", "", empty_df, empty_labels, "Frames analysed: 0"

    if MODEL is None or not CLASS_NAMES:
        return video_path, "### Model is not available.", "", empty_df, empty_labels, "Frames analysed: 0"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_path, "### Error: Could not open the uploaded video.\nPlease try a different file (e.g., MP4/AVI).", "", empty_df, empty_labels, "Frames analysed: 0"

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_sec = get_video_duration_seconds(frame_count, fps)

        if frame_count <= 0:
            return video_path, "### Error: No readable frames were found in this video.", "", empty_df, empty_labels, "Frames analysed: 0 | Video duration: unknown"

        indices = sample_frame_indices(frame_count, MAX_VIDEO_FRAMES)
        if len(indices) == 0:
            return video_path, "### Error: Could not select frames for analysis.", "", empty_df, empty_labels, "Frames analysed: 0"

        frame_probs = []
        sampled_count = 0

        # Extract selected frames (seek per index; max 20 frames keeps CPU/memory low)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            try:
                # Convert BGR (OpenCV) -> RGB (PIL/model preprocessing expects RGB)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                # Reuse the same preprocessing/normalization as training and image inference
                x = preprocess_image(pil_frame)
                probs = MODEL.predict(x, verbose=0)[0]
                frame_probs.append(np.asarray(probs, dtype=np.float32))
                sampled_count += 1
            except Exception:
                # Skip problematic frames but continue analyzing others
                continue

        if sampled_count == 0 or not frame_probs:
            return (
                video_path,
                "### Error: Frames could not be read or processed from this video.\nPlease try another video file.",
                "",
                empty_df,
                empty_labels,
                "Frames analysed: 0" + (f" | Video duration: {duration_sec:.2f}s" if duration_sec is not None else ""),
            )

        # Average class probabilities across sampled frames
        avg_probs = np.mean(np.stack(frame_probs, axis=0), axis=0)
        top_idx = int(np.argmax(avg_probs))
        pred_name = format_class_label(CLASS_NAMES[top_idx])
        avg_score = float(avg_probs[top_idx])

        prob_df = make_probability_dataframe(CLASS_NAMES, avg_probs)
        prob_dict = {row["Category"]: float(row["Probability"]) for _, row in prob_df.iterrows()}

        prediction_md = (
            f"## Dominant Predicted Category: **{pred_name}**\n"
            f"**Average prediction score (top class):** {avg_score:.2%}"
        )
        score_md = (
            f"Averaged over **{sampled_count}** sampled whole frame(s). "
            "This is a whole-frame prediction summary (not object-level)."
        )

        duration_text = f"{duration_sec:.2f} seconds" if duration_sec is not None else "Unknown"
        stats_md = (
            f"**Frames analysed:** {sampled_count} / {len(indices)} selected (max {MAX_VIDEO_FRAMES})  \n"
            f"**Video duration:** {duration_text}  \n"
            f"**Total frames (reported):** {frame_count}"
        )

        return video_path, prediction_md, score_md, prob_df, prob_dict, stats_md

    except Exception as e:
        return video_path, f"### Error during video analysis\n`{type(e).__name__}: {e}`", "", empty_df, empty_labels, "Frames analysed: 0"
    finally:
        cap.release()


def clear_video_all():
    """Reset all video classification UI components."""
    empty_df, empty_labels = empty_probability_outputs()
    return None, "### Upload a video to see the prediction.", "", empty_df, empty_labels, "Frames analysed: 0"


# -----------------------------
# UI
# -----------------------------

def build_app() -> gr.Blocks:
    """Create the Gradio UI with Image and Video Classification tabs."""
    title = "AI-Based Waste Classification System"
    description = (
        "Classify waste into six categories using the existing trained CNN. "
        "Use the tabs below for **image** or **video** (whole-frame) classification."
    )
    theme = gr.themes.Soft(primary_hue="green", neutral_hue="slate")

    with gr.Blocks(
        title=title,
        theme=theme,
        css="""
        .app-container {max-width: 1150px; margin: 0 auto;}
        .pred-box {border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #ffffff;}
        .muted {color: #475569;}
        .info-box {border: 1px solid #dbeafe; border-radius: 10px; padding: 10px; background: #f8fafc;}
        """
    ) as demo:
        with gr.Column(elem_classes=["app-container"]):
            gr.Markdown(f"# {title}")
            gr.Markdown(description)
            gr.Markdown(
                "**Categories:** Cardboard, Glass, Metal, Paper, Plastic, Trash  \n"
                "**Preprocessing (same as training):** RGB, resized to 96×96, normalized to [0, 1]."
            )

            if LOAD_ERROR:
                gr.Markdown(
                    f"> ⚠️ **Startup issue:** `{LOAD_ERROR}`\n\n"
                    "Ensure the trained model and class names files are available in `models/`."
                )
            if CV2_IMPORT_ERROR:
                gr.Markdown(
                    f"> ⚠️ **Video dependency issue:** OpenCV (`cv2`) is not available: `{CV2_IMPORT_ERROR}`\n\n"
                    "Image classification will still work if the model is loaded, but video classification requires OpenCV."
                )

            with gr.Tabs():
                # ---------------- Image Tab (preserved) ----------------
                with gr.TabItem("Image Classification"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5):
                            image_input = gr.Image(
                                label="Upload Waste Image (JPG / JPEG / PNG)",
                                type="pil",
                                sources=["upload", "clipboard"],
                                image_mode="RGB",
                                height=360,
                            )
                            with gr.Row():
                                image_predict_btn = gr.Button("Predict Image", variant="primary")
                                image_clear_btn = gr.Button("Clear / Reset", variant="secondary")

                        with gr.Column(scale=6):
                            image_preview = gr.Image(label="Uploaded Image", type="pil", interactive=False, height=360)
                            image_prediction_output = gr.Markdown(
                                "### Upload an image to see the prediction.",
                                elem_classes=["pred-box"],
                            )
                            image_confidence_output = gr.Markdown("", elem_classes=["muted"])

                    with gr.Row():
                        with gr.Column(scale=7):
                            image_prob_chart = gr.BarPlot(
                                label="Probability Chart (All 6 Categories)",
                                x="Category",
                                y="Probability",
                                title="Class Probabilities",
                                y_lim=[0, 1],
                                height=320,
                                tooltip=["Category", "Probability"],
                            )
                        with gr.Column(scale=5):
                            image_prob_labels = gr.Label(label="Class Probabilities", num_top_classes=6)

                # ---------------- Video Tab (new) ----------------
                with gr.TabItem("Video Classification"):
                    gr.Markdown(
                        """
<div class="info-box">
<b>Whole-frame video classification (not object detection/tracking):</b><br/>
This feature samples whole frames from the uploaded video (up to 20 frames, evenly spaced), runs the existing image CNN on each sampled frame,
and averages class probabilities to estimate the dominant waste category. It does <b>not</b> locate objects, draw bounding boxes, or track items across frames.
</div>
                        """
                    )
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5):
                            video_input = gr.Video(
                                label="Upload Video (MP4, AVI, and other supported formats)",
                                sources=["upload"],
                                height=360,
                            )
                            with gr.Row():
                                video_predict_btn = gr.Button("Analyze Video", variant="primary")
                                video_clear_btn = gr.Button("Clear / Reset", variant="secondary")

                        with gr.Column(scale=6):
                            video_prediction_output = gr.Markdown(
                                "### Upload a video to see the prediction.",
                                elem_classes=["pred-box"],
                            )
                            video_score_output = gr.Markdown("", elem_classes=["muted"])
                            video_stats_output = gr.Markdown("Frames analysed: 0", elem_classes=["muted"])

                    with gr.Row():
                        with gr.Column(scale=7):
                            video_prob_chart = gr.BarPlot(
                                label="Average Probability Distribution (All 6 Categories)",
                                x="Category",
                                y="Probability",
                                title="Averaged Class Probabilities Across Sampled Frames",
                                y_lim=[0, 1],
                                height=320,
                                tooltip=["Category", "Probability"],
                            )
                        with gr.Column(scale=5):
                            video_prob_labels = gr.Label(label="Average Class Probabilities", num_top_classes=6)

            gr.Markdown(
                "> **Disclaimer:** Predictions are experimental and may be incorrect. "
                "Lighting, background, motion blur, camera angle, occlusion, and mixed scenes can affect results. "
                "Video results are based on sampled whole frames and may not represent every moment/object in the clip."
            )

        # ---------------- Events: Image tab ----------------
        image_event_inputs = [image_input]
        image_event_outputs = [
            image_preview,
            image_prediction_output,
            image_confidence_output,
            image_prob_chart,
            image_prob_labels,
        ]
        image_predict_btn.click(fn=predict_waste, inputs=image_event_inputs, outputs=image_event_outputs)
        image_input.change(fn=predict_waste, inputs=image_event_inputs, outputs=image_event_outputs)
        image_clear_btn.click(
            fn=clear_image_all,
            inputs=None,
            outputs=[image_input] + image_event_outputs,
        )

        # ---------------- Events: Video tab ----------------
        video_event_inputs = [video_input]
        video_event_outputs = [
            video_input,  # keep/refresh video preview component
            video_prediction_output,
            video_score_output,
            video_prob_chart,
            video_prob_labels,
            video_stats_output,
        ]
        video_predict_btn.click(fn=predict_waste_video, inputs=video_event_inputs, outputs=video_event_outputs)
        video_clear_btn.click(fn=clear_video_all, inputs=None, outputs=video_event_outputs)

    return demo


if __name__ == "__main__":
    app = build_app()
    # Launch locally in browser (CPU-compatible). Set share=False for local-only.
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False)
