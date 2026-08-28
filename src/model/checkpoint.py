import logging
from pathlib import Path

import torch

from src.model.classifier import WhisperCommandClassifier

logger = logging.getLogger(__name__)

def save_checkpoint(path: str, model: WhisperCommandClassifier, label_to_idx: dict, idx_to_label: dict, whisper_model_name: str, freeze_encoder: bool, val_acc: float,
    epoch: int, fisher: dict | None = None, theta_star: dict | None = None,
    exemplars: dict | None = None, bias_correction: dict | None = None,
    seed: int | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if freeze_encoder:
        state = {"classifier_state_dict": model.classifier.state_dict()}
    else:
        logger.warning("Encoder was fine-tuned — saving full model state dict (checkpoint will be large).")
        state = {"model_state_dict": model.state_dict()}
    torch.save(
        {
            **state,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "whisper_model_name": whisper_model_name,
            "freeze_encoder": freeze_encoder,
            "head_hidden_dim": model.head_hidden_dim,
            "head_dropout": model.head_dropout,
            "val_acc": val_acc,
            "epoch": epoch,
            "fisher": fisher,
            "theta_star": theta_star,
            "exemplars": exemplars,
            "bias_correction": bias_correction,
            # Recorded so a sweep's results can be traced back to the run that produced
            # them rather than to whatever the source constant happened to say.
            "seed": seed,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint_extras(checkpoint_path: str) -> dict:
    """Read the metadata that load_checkpoint's tuple doesn't carry.

    Kept separate so load_checkpoint's return signature — unpacked positionally in
    several scripts — stays stable as new keys are added.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "exemplars": ckpt.get("exemplars"),
        "bias_correction": ckpt.get("bias_correction"),
        "val_acc": ckpt.get("val_acc"),
        "epoch": ckpt.get("epoch"),
        "seed": ckpt.get("seed"),
    }


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]
    whisper_model_name = ckpt["whisper_model_name"]
    freeze_encoder = ckpt.get("freeze_encoder", True)

    model = WhisperCommandClassifier(whisper_model_name, len(label_to_idx), freeze_encoder,
                                     head_hidden_dim=ckpt.get("head_hidden_dim", None), head_dropout=ckpt.get("head_dropout") or 0.0)
    if "classifier_state_dict" in ckpt:
        model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    else:
        logger.warning("Legacy checkpoint detected (full model_state_dict) — loading as-is.")
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    fisher = ckpt.get("fisher", None)
    theta_star = ckpt.get("theta_star", None)

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.1f}%)"
        + (" [has Fisher]" if fisher is not None else "")
    )
    return model, label_to_idx, idx_to_label, whisper_model_name, freeze_encoder, fisher, theta_star