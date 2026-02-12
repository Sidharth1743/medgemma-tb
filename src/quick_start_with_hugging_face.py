from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, SiglipVisionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official MedSigLIP quick-start flow using local Hugging Face files."
    )
    parser.add_argument("--model-dir", default="medsiglip", help="Local model directory.")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more input image paths.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[
            "a photo of an arm with no rash",
            "a photo of an arm with a rash",
            "a photo of a leg with no rash",
            "a photo of a leg with a rash",
        ],
        help="Candidate labels for zero-shot classification.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_weights(model_dir: Path) -> None:
    has_weights = any(model_dir.glob("*.safetensors")) or any(
        model_dir.glob("pytorch_model*.bin")
    )
    if has_weights:
        return
    root_weights = PROJECT_ROOT / "model.safetensors"
    if root_weights.exists():
        (model_dir / "model.safetensors").symlink_to(root_weights)
        print(f"Created symlink: {model_dir / 'model.safetensors'} -> {root_weights}")
        return
    raise FileNotFoundError(f"No model weights found in {model_dir}")


def load_images(image_paths: list[Path]) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def verify_auto_resize(processor: AutoProcessor, imgs: list[Image.Image]) -> None:
    image_processor = processor.image_processor
    do_resize = bool(getattr(image_processor, "do_resize", False))
    size = getattr(image_processor, "size", {})
    expected_h = size.get("height") if isinstance(size, dict) else None
    expected_w = size.get("width") if isinstance(size, dict) else None

    print(
        f"Processor resize config: do_resize={do_resize}, expected_size=({expected_h}, {expected_w})"
    )

    check_batch = processor(images=imgs, return_tensors="pt")
    pixel_values = check_batch["pixel_values"]
    actual_h, actual_w = pixel_values.shape[-2], pixel_values.shape[-1]
    print(f"Preprocessed tensor size: ({actual_h}, {actual_w})")

    if do_resize and expected_h == actual_h and expected_w == actual_w:
        print("Auto-resize verification: PASS")
    else:
        print("Auto-resize verification: CHECK MANUALLY")


def main() -> None:
    args = parse_args()
    model_dir = resolve_path(args.model_dir)
    ensure_weights(model_dir)

    image_paths = [resolve_path(p) for p in args.images]
    imgs = load_images(image_paths)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device={device}, dtype={dtype}, model_dir={model_dir}")

    zero_shot_model = AutoModel.from_pretrained(model_dir, dtype=dtype).to(device)
    processor = AutoProcessor.from_pretrained(model_dir)
    verify_auto_resize(processor, imgs)

    inputs = processor(
        text=args.labels, images=imgs, padding="max_length", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = zero_shot_model(**inputs)

    bests = np.argmax(outputs.logits_per_image.detach().float().cpu().numpy(), axis=1)
    for index, best in enumerate(bests):
        print(f"Best label for image {index}: '{args.labels[best]}'")

    vision_model = SiglipVisionModel.from_pretrained(model_dir, dtype=dtype).to(device)
    vision_inputs = processor(images=imgs, padding="max_length", return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        vision_outputs = vision_model(**vision_inputs)

    embeddings = vision_outputs.pooler_output
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    print(f"Embedding shape: {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
