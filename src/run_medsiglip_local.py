from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModel,
    SiglipImageProcessor,
    SiglipProcessor,
    SiglipTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local MedSigLIP zero-shot image classification."
    )
    parser.add_argument(
        "--model-dir",
        default="medsiglip",
        help="Directory containing MedSigLIP config/tokenizer/processor files.",
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Candidate text labels or prompts.",
    )
    return parser.parse_args()


def resolve_model_dir(model_dir_arg: str) -> Path:
    model_dir = Path(model_dir_arg)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_dir = model_dir.resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    return model_dir


def ensure_local_weights(model_dir: Path) -> None:
    has_weights = any(model_dir.glob("*.safetensors")) or any(
        model_dir.glob("pytorch_model*.bin")
    )
    if has_weights:
        return

    root_weights = PROJECT_ROOT / "model.safetensors"
    target_weights = model_dir / "model.safetensors"
    if root_weights.exists() and not target_weights.exists():
        try:
            target_weights.symlink_to(root_weights)
            print(f"Created symlink: {target_weights} -> {root_weights}")
            return
        except OSError:
            pass

    raise FileNotFoundError(
        "No model weights found in model dir. Place `model.safetensors` inside "
        f"`{model_dir}` (or keep it at `{root_weights}` and rerun)."
    )


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    ensure_local_weights(model_dir)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device={device}, dtype={dtype}, model_dir={model_dir}")
    try:
        model = AutoModel.from_pretrained(
            model_dir, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
    except TypeError:
        model = AutoModel.from_pretrained(
            model_dir, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)

    try:
        image_processor = SiglipImageProcessor.from_pretrained(model_dir)
        tokenizer = SiglipTokenizer.from_pretrained(model_dir)
        processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    except ImportError as exc:
        raise ImportError(
            "Missing dependency for tokenizer. Install `sentencepiece` and `protobuf`."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=args.labels,
        images=[image],
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits_per_image[0], dim=0)

    ranked = sorted(zip(args.labels, probs.tolist()), key=lambda x: x[1], reverse=True)
    print("\nPredictions (highest first):")
    for label, score in ranked:
        print(f"{score:.2%}\t{label}")


if __name__ == "__main__":
    main()
