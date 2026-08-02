import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)

_FISHER_MAX_BATCHES = 100


def output_layer_param_names(model) -> tuple[str, str]:
    """Keys of the output layer's weight/bias within model.classifier.named_parameters()."""
    if isinstance(model.classifier, nn.Sequential):
        last = len(model.classifier) - 1
        return f"{last}.weight", f"{last}.bias"
    return "weight", "bias"


def configure_head_training(model, train_hidden_layer: bool) -> list:
    """Select which head parameters the continual scripts optimise, freezing the rest.

    train_hidden_layer=False keeps the base model's representation fixed and updates only
    the output layer. True also updates the shared hidden layers — which is what EWC and
    distillation exist to make safe, but which nothing protects if those mechanisms don't
    cover the hidden parameters.

    The unselected parameters are frozen explicitly rather than merely omitted from the
    optimiser, so gradients don't accumulate in buffers nothing ever reads.
    """
    out_layer = model.output_layer
    if train_hidden_layer:
        for p in model.classifier.parameters():
            p.requires_grad_(True)
        trainable = list(model.classifier.parameters())
    else:
        for p in model.classifier.parameters():
            p.requires_grad_(False)
        out_layer.weight.requires_grad_(True)
        out_layer.bias.requires_grad_(True)
        trainable = [out_layer.weight, out_layer.bias]

    n_params = sum(p.numel() for p in trainable)
    logger.info("Training %d head parameter tensors (%d values) — hidden layer %s.",
                len(trainable), n_params, "trainable" if train_hidden_layer else "frozen")
    return trainable


def compute_fisher_diagonal(model, train_loader, device) -> tuple[dict, dict]:
    """Diagonal Fisher Information Matrix over the classifier head.

    Keyed by parameter name within model.classifier, so a plain single-Linear head yields
    {"weight", "bias"} — the same layout earlier checkpoints already use. A head with a
    hidden layer additionally yields its entries, letting EWC protect them when
    TRAIN_HIDDEN_LAYER is on.
    """
    model.eval()
    named = {n: p for n, p in model.classifier.named_parameters() if p.requires_grad}
    if not named:
        logger.warning("compute_fisher_diagonal: no trainable head parameters.")
        return {}, {}
    fisher = {n: torch.zeros_like(p) for n, p in named.items()}

    n_batches = 0
    for mels, _, n_frames in train_loader:
        if n_batches >= _FISHER_MAX_BATCHES:
            break
        mels     = mels.to(device)
        n_frames = n_frames.to(device)

        for p in named.values():
            p.grad = None

        logits    = model(mels, n_frames)
        log_probs = F.log_softmax(logits, dim=1)
        predicted = logits.argmax(dim=1)
        loss      = F.nll_loss(log_probs, predicted)
        loss.backward()

        for name, p in named.items():
            if p.grad is not None:
                fisher[name] += p.grad ** 2

        for p in named.values():
            p.grad = None

        n_batches += 1

    theta_star = {n: p.detach().clone() for n, p in named.items()}

    if n_batches == 0:
        logger.warning("compute_fisher_diagonal: train_loader was empty, returning zero Fisher.")
        return fisher, theta_star

    for name in fisher:
        fisher[name] /= n_batches

    logger.info("Fisher diagonal computed over %d batches for: %s", n_batches, sorted(fisher))
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