import logging
import sys
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.checkpoint import load_checkpoint, save_checkpoint
from src.model.classifier import WhisperCommandClassifier
from src.training.trainer import compute_fisher_diagonal, configure_head_training, output_layer_param_names
from src.utils.seed import resolve_seed, set_seed
from scripts.continual_train import (LR, build_dataloaders, collect_files, expand_classifier, run_data_generation)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/BASE.pth"
HEAD_DROPOUT = None
SEED = resolve_seed()
TRAIN_HIDDEN_LAYER = False

# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt_new_command() -> str:
    print("\n=== New Command Setup ===")
    label = input("Label key (e.g. Open_Spotify): ").strip()
    while not label:
        label = input("Label cannot be empty. Label key: ").strip()
    return label


def prompt_training_config() -> tuple[str, int, float]:
    model_name = input("\nEnter name for the new checkpoint: ").strip()
    while True:
        try:
            epochs = int(input("Number of epochs: "))
            break
        except ValueError:
            print("Please enter a valid number.")
    while True:
        try:
            ewc_lambda = float(input("EWC lambda [400.0]: ").strip() or "400.0")
            break
        except ValueError:
            print("Please enter a valid number.")
    return model_name, epochs, ewc_lambda

# ---------------------------------------------------------------------------
# EWC helpers
# ---------------------------------------------------------------------------

def pad_fisher_and_theta(fisher: dict, theta_star: dict, out_keys: tuple[str, str]) -> tuple[dict, dict]:
    """Append a zero row to Fisher and theta* for the new (N+1-th) head row.

    Zero Fisher means EWC places no penalty on the new row — it learns freely. Hidden
    layer entries don't change shape when a class is added, so they pass through as-is.
    """
    w_key, b_key = out_keys
    fisher, theta_star = dict(fisher), dict(theta_star)
    D = fisher[w_key].shape[1]
    dev = fisher[w_key].device

    fisher[w_key] = torch.cat([fisher[w_key], torch.zeros(1, D, device=dev)])
    fisher[b_key] = torch.cat([fisher[b_key], torch.zeros(1, device=dev)])
    theta_star[w_key] = torch.cat([theta_star[w_key], torch.zeros(1, D, device=dev)])
    theta_star[b_key] = torch.cat([theta_star[b_key], torch.zeros(1, device=dev)])
    return fisher, theta_star

def accumulate_and_resave(checkpoint_path: str, train_loader, device: torch.device, label_to_idx: dict, idx_to_label: dict,
    whisper_model_name: str, old_fisher: dict) -> None:
    """Load the best checkpoint, compute Fisher on new training data, accumulate, re-save."""
    logger.info("Accumulating Fisher on best checkpoint weights...")

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    val_acc = raw["val_acc"]
    epoch   = raw["epoch"]

    model, _, _, _, _, _, _ = load_checkpoint(checkpoint_path, str(device))
    model.to(device)

    new_fisher, new_theta_star = compute_fisher_diagonal(model, train_loader, device)

    accumulated = {
        name: old_fisher[name].cpu() + new_fisher[name].cpu()
        for name in new_fisher if name in old_fisher
    }
    final_theta_star = {k: v.cpu() for k, v in new_theta_star.items()}

    save_checkpoint(
        checkpoint_path, model, label_to_idx, idx_to_label,
        whisper_model_name, freeze_encoder=True,
        val_acc=val_acc, epoch=epoch,
        fisher=accumulated,
        theta_star=final_theta_star,
        seed=SEED,
    )
    logger.info("Fisher accumulated and checkpoint re-saved → %s", checkpoint_path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ewc(model: WhisperCommandClassifier, train_loader, val_loader, device: torch.device, epochs: int,
    lr: float, checkpoint_path: str, label_to_idx: dict, idx_to_label: dict, whisper_model_name: str, fisher: dict,
    theta_star: dict, ewc_lambda: float, train_hidden_layer: bool = TRAIN_HIDDEN_LAYER) -> None:
    """Train/val loop with EWC loss, saving best checkpoint by val accuracy."""
    fisher_d     = {k: v.to(device) for k, v in fisher.items()}
    theta_star_d = {k: v.to(device) for k, v in theta_star.items()}

    optimizer = torch.optim.AdamW(configure_head_training(model, train_hidden_layer), lr=lr)
    criterion    = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # Penalise every parameter that can actually move and has a Fisher entry. Frozen
    # parameters would only contribute a constant.
    named        = dict(model.classifier.named_parameters())
    penalty_keys = [k for k in fisher_d if k in named and named[k].requires_grad]
    logger.info("EWC penalty applied to: %s", penalty_keys)
    unprotected  = [k for k, p in named.items() if p.requires_grad and k not in fisher_d]
    if unprotected:
        logger.warning("Trainable but NOT covered by Fisher (will drift freely): %s", unprotected)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        t_loss = t_correct = t_total = 0
        for mels, label_idxs, n_frames in train_loader:
            mels       = mels.to(device)
            label_idxs = label_idxs.to(device)
            n_frames   = n_frames.to(device)

            optimizer.zero_grad()
            logits    = model(mels, n_frames)
            task_loss = criterion(logits, label_idxs)

            ewc_penalty = sum(
                (fisher_d[k] * (named[k] - theta_star_d[k]) ** 2).sum() for k in penalty_keys
            )

            loss = task_loss + (ewc_lambda / 2) * ewc_penalty
            loss.backward()
            optimizer.step()

            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == label_idxs).sum().item()
            t_total   += label_idxs.size(0)

        # --- Validate ---
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for mels, label_idxs, n_frames in val_loader:
                mels       = mels.to(device)
                label_idxs = label_idxs.to(device)
                n_frames   = n_frames.to(device)
                logits     = model(mels, n_frames)
                loss       = criterion(logits, label_idxs)
                v_loss    += loss.item()
                v_correct += (logits.argmax(1) == label_idxs).sum().item()
                v_total   += label_idxs.size(0)

        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        logger.info(
            "Epoch %02d/%02d | Train %.4f / %.1f%% | Val %.4f / %.1f%%",
            epoch + 1, epochs,
            t_loss / len(train_loader), t_acc,
            v_loss / len(val_loader), v_acc,
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint(
                checkpoint_path, model, label_to_idx, idx_to_label,
                whisper_model_name, freeze_encoder=True,
                val_acc=v_acc, epoch=epoch + 1,
                fisher={k: v.cpu() for k, v in fisher_d.items()},
                theta_star={k: v.cpu() for k, v in theta_star_d.items()},
                seed=SEED,
            )
            logger.info("  → Best checkpoint saved (val_acc=%.1f%%)", v_acc)

    p = Path(checkpoint_path)
    le_path = str(p.with_name(f"{p.stem}_LE{p.suffix}"))
    save_checkpoint(
        le_path, model, label_to_idx, idx_to_label,
        whisper_model_name, freeze_encoder=True,
        val_acc=v_acc, epoch=epochs,
        fisher={k: v.cpu() for k, v in fisher_d.items()},
        theta_star={k: v.cpu() for k, v in theta_star_d.items()},
        seed=SEED,
    )
    logger.info("  → Last-epoch checkpoint saved → %s (val_acc=%.1f%%)", le_path, v_acc)

    logger.info("Training complete. Best val accuracy: %.1f%%", best_val_acc)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    data_dir = Path("data")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load BASE checkpoint
    logger.info("Loading checkpoint: %s", CHECKPOINT_PATH)
    old_model, label_to_idx, idx_to_label, whisper_model_name, _, fisher, theta_star = (
        load_checkpoint(CHECKPOINT_PATH, str(device))
    )
    if fisher is None or theta_star is None:
        logger.error(
            "Checkpoint has no Fisher data. Re-train with train.py "
            "(Fisher computation is on by default — omit --no-fisher)."
        )
        sys.exit(1)

    n_old = len(label_to_idx)
    logger.info("Existing classes (%d): %s", n_old, list(label_to_idx.keys()))

    # 2. Prompt for new command label and training config
    new_label = prompt_new_command()
    if new_label in label_to_idx:
        logger.error("Label '%s' already exists in this checkpoint. Aborting.", new_label)
        sys.exit(1)

    model_name, epochs, ewc_lambda = prompt_training_config()
    checkpoint_out = f"models/{date.today()}_{model_name}.pth"

    # 3. Generate data unless the label directory already exists
    label_dir = data_dir / new_label
    if label_dir.exists():
        logger.info("Data directory '%s' already exists, skipping generation.", label_dir)
    else:
        run_data_generation(new_label, data_dir)

    # 4. Expand label maps (new label appended at index N, old indices unchanged)
    new_idx = n_old
    label_to_idx[new_label] = new_idx
    idx_to_label[new_idx]   = new_label
    logger.info("New label '%s' assigned index %d", new_label, new_idx)

    # 5. Build expanded model
    logger.info("Building %d-class model (was %d)...", n_old + 1, n_old)
    model = expand_classifier(old_model, whisper_model_name, n_old + 1, device, head_dropout=HEAD_DROPOUT)
    del old_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 6. Pad Fisher and theta* for the new head row
    out_keys = output_layer_param_names(model)
    if out_keys[0] not in fisher:
        logger.error(
            "Checkpoint's Fisher has keys %s but this head's output layer is %s. "
            "Re-train the base with train.py so Fisher matches the architecture.",
            sorted(fisher), list(out_keys),
        )
        sys.exit(1)
    if TRAIN_HIDDEN_LAYER:
        hidden_keys = [n for n, _ in model.classifier.named_parameters() if n not in out_keys]
        missing = [k for k in hidden_keys if k not in fisher]
        if missing:
            logger.error(
                "TRAIN_HIDDEN_LAYER is on but the checkpoint has no Fisher for %s, so those "
                "weights would drift with no EWC penalty at all. Re-train the base with "
                "train.py (which computes Fisher over the whole head), or set "
                "TRAIN_HIDDEN_LAYER = False.",
                missing,
            )
            sys.exit(1)

    fisher, theta_star = pad_fisher_and_theta(fisher, theta_star, out_keys)
    logger.info("Fisher and theta* padded to %d classes.", n_old + 1)

    # 7. Build dataloaders (new command data only — EWC penalty handles forgetting)
    file_paths, labels = collect_files(new_label, data_dir)
    train_loader, val_loader = build_dataloaders(file_paths, labels, label_to_idx, model.n_mels, seed=SEED)
    logger.info("Train: %d  Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    # 8. Train with EWC loss
    logger.info(
        "Training '%s' for %d epochs with EWC λ=%.1f → %s",
        new_label, epochs, ewc_lambda, checkpoint_out,
    )
    train_ewc(
        model, train_loader, val_loader, device,
        epochs, LR, checkpoint_out,
        label_to_idx, idx_to_label, whisper_model_name,
        fisher, theta_star, ewc_lambda,
        train_hidden_layer=TRAIN_HIDDEN_LAYER,
    )

    # 9. Accumulate Fisher on new training data and re-save best checkpoint
    accumulate_and_resave(
        checkpoint_out, train_loader, device,
        label_to_idx, idx_to_label, whisper_model_name,
        old_fisher=fisher,
    )


if __name__ == "__main__":
    main()