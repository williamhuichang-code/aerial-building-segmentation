"""
Building Roof Segmentation from Aerial Imagery: interactive demo.

Run locally:
    set MODEL_PATH=path\\to\\optimised_config.pt     (Windows)
    python app.py
Then open http://127.0.0.1:7860
"""
from pathlib import Path

import gradio as gr
import torch

from inference import GroundTruth, draw_predictions, load_model, load_rgb, predict

HERE = Path(__file__).parent
SAMPLES = sorted((HERE / "samples").glob("*.png"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = load_model(DEVICE)
GT = GroundTruth(str(HERE / "samples" / "ground_truth.json"))


def segment(image_path, score_threshold):
    if image_path is None:
        return None, None, "Choose a sample tile or upload an aerial image."
    img = load_rgb(image_path)
    pred = predict(MODEL, img, score_threshold, DEVICE)
    pred_img = draw_predictions(img, pred, score_threshold)

    n_pred = len(pred["scores"])
    summary = f"**{n_pred} building{'s' if n_pred != 1 else ''} detected** (confidence ≥ {score_threshold:.2f})"
    if GT.has(image_path):
        gt_img = GT.draw(img, image_path)
        summary += f"  \nGround truth (LINZ building outlines): **{GT.count(image_path)}**"
    else:
        gt_img = None
        summary += "  \nNo ground truth available for uploaded images."
    return pred_img, gt_img, summary


DESCRIPTION = """
Mask R-CNN (ResNet-50 FPN) fine-tuned to outline building roofs in Christchurch aerial imagery,
developed in a research project with **Earth Sciences New Zealand** to support hazard-risk exposure modelling.
Held-out test performance: **mAP 0.51**, **AP50 0.86**.

Pick a sample tile from around the University of Canterbury, or upload your own top-down aerial image.
Predictions are shown in **purple** (bluer = less confident); ground truth in **green**.
"""

with gr.Blocks(title="Building Roof Segmentation") as demo:
    gr.Markdown("# Building Roof Segmentation from Aerial Imagery")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="filepath", label="Aerial image", height=360)
            thr = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="Confidence threshold")
            btn = gr.Button("Segment buildings", variant="primary")
            gr.Examples(examples=[[str(p)] for p in SAMPLES], inputs=[inp], label="Sample tiles (around UC, Christchurch)")
        with gr.Column(scale=2):
            info = gr.Markdown()
            with gr.Row():
                out_pred = gr.Image(label="Model prediction", height=420)
                out_gt = gr.Image(label="Ground truth", height=420)
    gr.Markdown(
        "Imagery and building outlines: Land Information New Zealand (LINZ), CC BY 4.0. "
        "Model trained by William Hui Chang."
    )
    btn.click(segment, inputs=[inp, thr], outputs=[out_pred, out_gt, info])
    inp.change(segment, inputs=[inp, thr], outputs=[out_pred, out_gt, info])
    thr.release(segment, inputs=[inp, thr], outputs=[out_pred, out_gt, info])

if __name__ == "__main__":
    demo.launch()
