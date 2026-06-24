import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)

_FISHER_MAX_BATCHES = 100


def compute_fisher_diagonal(model, train_loader, device) -> tuple[dict, dict]:
    """Compute diagonal Fisher Information Matrix over the classifier head."""
    model.eval()
    out_layer = model.output_layer
    fisher = {
        "weight": torch.zeros_like(out_layer.weight),
        "bias":   torch.zeros_like(out_layer.bias),
    }

    n_batches = 0
    for mels, _, n_frames in train_loader:
        if n_batches >= _FISHER_MAX_BATCHES:
            break
        mels     = mels.to(device)
        n_frames = n_frames.to(device)

        out_layer.weight.grad = None
        out_layer.bias.grad   = None

        logits    = model(mels, n_frames)
        log_probs = F.log_softmax(logits, dim=1)
        predicted = logits.argmax(dim=1)
        loss      = F.nll_loss(log_probs, predicted)
        loss.backward()

        fisher["weight"] += out_layer.weight.grad ** 2
        fisher["bias"]   += out_layer.bias.grad   ** 2

        out_layer.weight.grad = None
        out_layer.bias.grad   = None

        n_batches += 1

    if n_batches == 0:
        logger.warning("compute_fisher_diagonal: train_loader was empty, returning zero Fisher.")
        return fisher, {"weight": out_layer.weight.detach().clone(),
                        "bias":   out_layer.bias.detach().clone()}

    fisher["weight"] /= n_batches
    fisher["bias"]   /= n_batches

    theta_star = {
        "weight": out_layer.weight.detach().clone(),
        "bias":   out_layer.bias.detach().clone(),
    }

    logger.info("Fisher diagonal computed over %d batches.", n_batches)
    return fisher, theta_star


def train_model(model, train_loader, val_loader, device, epochs: int, lr: float, checkpoint_path: str,
                label_to_idx: dict, idx_to_label: dict, whisper_model_name: str, freeze_encoder: bool,
                compute_fisher: bool = True) -> None:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        t_loss = t_correct = t_total = 0
        for mels, label_idxs, n_frames in train_loader:
            mels = mels.to(device)
            label_idxs = label_idxs.to(device)
            n_frames = n_frames.to(device)

            optimizer.zero_grad()
            logits = model(mels, n_frames)
            loss = criterion(logits, label_idxs)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_correct += (logits.argmax(1) == label_idxs).sum().item()
            t_total += label_idxs.size(0)

        # --- Validate ---
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for mels, label_idxs, n_frames in val_loader:
                mels = mels.to(device)
                label_idxs = label_idxs.to(device)
                n_frames = n_frames.to(device)

                logits = model(mels, n_frames)
                loss = criterion(logits, label_idxs)
                v_loss += loss.item()
                v_correct += (logits.argmax(1) == label_idxs).sum().item()
                v_total += label_idxs.size(0)

        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        logger.info(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train {t_loss / len(train_loader):.4f} / {t_acc:.1f}% | "
            f"Val {v_loss / len(val_loader):.4f} / {v_acc:.1f}%"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            fisher, theta_star = (compute_fisher_diagonal(model, train_loader, device)
                                  if compute_fisher else (None, None))
            save_checkpoint(
                checkpoint_path, model, label_to_idx, idx_to_label,
                whisper_model_name, freeze_encoder, v_acc, epoch + 1,
                fisher=fisher, theta_star=theta_star,
            )
            logger.info(f"  → Best checkpoint saved (val_acc={v_acc:.1f}%)"
                        + (" [Fisher computed]" if compute_fisher else ""))

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.1f}%")