import copy
import logging
import random
import sys
from datetime import date
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.checkpoint import load_checkpoint, load_checkpoint_extras, save_checkpoint
from src.model.classifier import WhisperCommandClassifier
from src.training.dataset import CommandDataset
from src.training.distillation import (fit_bias_correction, fit_bias_vector, fold_bias_correction,
                                       fold_bias_vector, icarl_distillation_loss)
from src.training.exemplars import partition_old_classes, resolve_exemplars, select_exemplars_herding
from src.training.trainer import configure_head_training
from src.utils.seed import dataloader_generator, resolve_seed, set_seed
from scripts.continual_train import (BATCH_SIZE, LR, collect_files, expand_classifier, run_data_generation)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/BASE.pth"
HEAD_DROPOUT = None
SEED = resolve_seed()
TRAIN_HIDDEN_LAYER = False

REPLAY_CE_WEIGHT = 1.0

# Exemplars kept per class for replay + distillation.
EXEMPLARS_PER_CLASS = 20
BIC_VAL_PER_CLASS = 5
OLD_VAL_SAMPLES_PER_CLASS = 15

BIC_EPOCHS = 200
BIC_LR = 1e-3

BIC_MODE_SCALAR = "scalar"
BIC_MODE_VECTOR = "vector"
BIC_MODE_VECTOR_ALPHA = "vector_alpha"


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt_new_command() -> str:
    print("\n=== New Command Setup ===")
    label = input("Label key (e.g. Open_Spotify): ").strip()
    while not label:
        label = input("Label cannot be empty. Label key: ").strip()
    return label


def prompt_bias_correction_mode() -> str:
    print("\nBias correction mode:")
    print("  [1] Traditional BiC — scalar α/β on the new-class logits")
    print("  [2] Bias-only vector scaling — one additive offset per class")
    print("  [3] Vector scaling with a shared α — α·logits + b, α over every class")
    while True:
        choice = input("Selection [1]: ").strip() or "1"
        if choice == "1":
            return BIC_MODE_SCALAR
        if choice == "2":
            return BIC_MODE_VECTOR
        if choice == "3":
            return BIC_MODE_VECTOR_ALPHA
        print("Please enter 1, 2 or 3.")


def prompt_training_config() -> tuple[str, int, str]:
    model_name = input("\nEnter name for the new checkpoint: ").strip()
    while True:
        try:
            epochs = int(input("Number of epochs: "))
            break
        except ValueError:
            print("Please enter a valid number.")
    bic_mode = prompt_bias_correction_mode()
    return model_name, epochs, bic_mode


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _loader(paths: list[str], labels: list[str], label_to_idx: dict, n_mels: int,
            augment: bool, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    ds = CommandDataset(paths, labels, label_to_idx, n_mels, augment=augment)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, generator=generator)


def build_loaders(exemplars: dict, new_label: str, new_files: list[str], data_dir: Path,
                  label_to_idx: dict, n_mels: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Partition everything three ways — see partition_old_classes for the old-class side.

    The new class is split 80/20; the bias-correction clips are carved out of the val
    side so they never touch stage-1 training.

    Every split is drawn from `seed`. They used to be fixed - random_state=42 plus a
    fixed slice of the herding order - which left a seed sweep re-training on
    byte-identical data in every run.
    """
    (ex_train, ex_train_lbl), (ex_bic, ex_bic_lbl), (old_val, old_val_lbl) = partition_old_classes(
        exemplars, data_dir, BIC_VAL_PER_CLASS, OLD_VAL_SAMPLES_PER_CLASS, rng=random.Random(seed)
    )

    new_train, new_held = train_test_split(new_files, test_size=0.2, random_state=seed)
    new_bic, new_val = new_held[:BIC_VAL_PER_CLASS], new_held[BIC_VAL_PER_CLASS:]

    train_loader = _loader(
        ex_train + new_train, ex_train_lbl + [new_label] * len(new_train),
        label_to_idx, n_mels, augment=True, shuffle=True, generator=dataloader_generator(seed),
    )
    bic_loader = _loader(
        ex_bic + new_bic, ex_bic_lbl + [new_label] * len(new_bic),
        label_to_idx, n_mels, augment=False, shuffle=False,
    )
    val_loader = _loader(
        old_val + new_val, old_val_lbl + [new_label] * len(new_val),
        label_to_idx, n_mels, augment=False, shuffle=False,
    )

    logger.info("Train: %d (%d exemplars + %d new)", len(ex_train) + len(new_train), len(ex_train), len(new_train))
    logger.info("Bias-correction: %d (%d old + %d new)", len(ex_bic) + len(new_bic), len(ex_bic), len(new_bic))
    logger.info("Val: %d (%d old + %d new)", len(old_val) + len(new_val), len(old_val), len(new_val))
    if not ex_bic or not new_bic:
        logger.warning("Bias-correction set is missing one side — α/β will be poorly determined.")

    return train_loader, bic_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "n/a"


@torch.no_grad()
def evaluate(model: WhisperCommandClassifier, loader: DataLoader, device: torch.device, n_old: int,
             teacher_head: torch.nn.Module | None = None) -> dict:
    """Accuracy overall / on new classes / on old classes, plus distillation loss if a
    teacher is supplied."""
    model.eval()
    loss_sum = 0.0
    correct = total = new_correct = new_total = 0

    for mels, label_idxs, n_frames in loader:
        mels = mels.to(device)
        label_idxs = label_idxs.to(device)
        n_frames = n_frames.to(device)

        pooled = model.pooled_features(mels, n_frames)
        logits = model.classifier(pooled)
        if teacher_head is not None:
            loss, _, _, _ = icarl_distillation_loss(
                logits, teacher_head(pooled), label_idxs, n_old, REPLAY_CE_WEIGHT
            )
            loss_sum += loss.item()

        hit = logits.argmax(1) == label_idxs
        correct += hit.sum().item()
        total += label_idxs.size(0)
        new_mask = label_idxs >= n_old
        new_correct += hit[new_mask].sum().item()
        new_total += new_mask.sum().item()

    old_total = total - new_total
    return {
        "loss": loss_sum / len(loader) if len(loader) else 0.0,
        "acc": 100 * correct / total if total else 0.0,
        "new_acc": 100 * new_correct / new_total if new_total else None,
        "old_acc": 100 * (correct - new_correct) / old_total if old_total else None,
    }


# ---------------------------------------------------------------------------
# Stage 1: distillation training
# ---------------------------------------------------------------------------

def train_distillation(model: WhisperCommandClassifier, teacher_head: torch.nn.Module, train_loader: DataLoader,
                       val_loader: DataLoader, device: torch.device, epochs: int, n_old: int, checkpoint_path: str,
                       le_path: str, label_to_idx: dict, idx_to_label: dict, whisper_model_name: str,
                       freeze_encoder: bool, train_hidden_layer: bool = TRAIN_HIDDEN_LAYER) -> None:
    """Train the head under the iCaRL loss, saving the best and last-epoch checkpoints.

    Unlike continual_train.py the old head rows are fully trainable — holding them still
    would defeat the point, since the distillation term is what now protects them. The
    same argument extends to the hidden layer when train_hidden_layer is set.
    """
    optimizer = torch.optim.AdamW(configure_head_training(model, train_hidden_layer), lr=LR)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        t_loss = t_distill = t_class = t_replay = 0.0
        t_correct = t_total = 0
        for mels, label_idxs, n_frames in train_loader:
            mels = mels.to(device)
            label_idxs = label_idxs.to(device)
            n_frames = n_frames.to(device)

            optimizer.zero_grad()
            # One encoder pass feeds both heads, so the teacher sees exactly the same
            # SpecAugment'd input as the student.
            pooled = model.pooled_features(mels, n_frames)
            logits = model.classifier(pooled)
            with torch.no_grad():
                teacher_logits = teacher_head(pooled)

            loss, distill, classification, replay = icarl_distillation_loss(
                logits, teacher_logits, label_idxs, n_old, REPLAY_CE_WEIGHT
            )
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_distill += distill.item()
            t_class += classification.item()
            t_replay += replay.item()
            t_correct += (logits.argmax(1) == label_idxs).sum().item()
            t_total += label_idxs.size(0)

        # --- Validate ---
        val = evaluate(model, val_loader, device, n_old, teacher_head)

        n_batches = len(train_loader)
        logger.info(
            "Epoch %02d/%02d | Train %.4f (d %.4f / c %.4f / r %.4f) / %.1f%% | Val %.4f / %.1f%% | new-cls %s | old-cls %s",
            epoch + 1, epochs,
            t_loss / n_batches, t_distill / n_batches, t_class / n_batches, t_replay / n_batches,
            100 * t_correct / t_total,
            val["loss"], val["acc"], _pct(val["new_acc"]), _pct(val["old_acc"]),
        )

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            save_checkpoint(
                checkpoint_path, model, label_to_idx, idx_to_label,
                whisper_model_name, freeze_encoder, val["acc"], epoch + 1,
                seed=SEED,
            )
            logger.info("  → Best checkpoint saved (val_acc=%.1f%%)", val["acc"])

    save_checkpoint(
        le_path, model, label_to_idx, idx_to_label,
        whisper_model_name, freeze_encoder, val["acc"], epochs,
        seed=SEED,
    )
    logger.info("  → Last-epoch checkpoint saved → %s (val_acc=%.1f%%)", le_path, val["acc"])
    logger.info("Stage 1 complete. Best val accuracy: %.1f%%", best_val_acc)


# ---------------------------------------------------------------------------
# Stage 2: bias correction
# ---------------------------------------------------------------------------

def reload_head(model: WhisperCommandClassifier, checkpoint_path: str, device: torch.device) -> None:
    """Restore a saved checkpoint's head into an existing model.

    Only the head ever changes here, so this avoids re-loading the Whisper encoder — a
    full load_checkpoint per stage-2 pass would cost more than the correction itself.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "classifier_state_dict" in ckpt:
        model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)


def _format_bias_vector(bias_vector: list[float], idx_to_label: dict) -> str:
    return "  ".join(f"{idx_to_label.get(i, i)}={v:+.4f}" for i, v in enumerate(bias_vector))


def apply_bias_correction(model: WhisperCommandClassifier, checkpoint_path: str, bic_loader: DataLoader,
                          val_loader: DataLoader, device: torch.device, n_old: int, label_to_idx: dict,
                          idx_to_label: dict, whisper_model_name: str, freeze_encoder: bool,
                          exemplars: dict, mode: str = BIC_MODE_SCALAR) -> None:
    """Fit the calibration on the held-out balanced set, fold it into the head, and re-save.

    All three modes leave the network an ordinary classifier, so the choice is invisible to
    everything downstream; it is recorded in the checkpoint only so a result can be traced
    back to the arm that produced it.
    """
    logger.info("--- BiC stage 2 (%s): %s ---", mode, checkpoint_path)
    reload_head(model, checkpoint_path, device)
    extras = load_checkpoint_extras(checkpoint_path)

    before = evaluate(model, val_loader, device, n_old)

    if mode in (BIC_MODE_VECTOR, BIC_MODE_VECTOR_ALPHA):
        fit_alpha = mode == BIC_MODE_VECTOR_ALPHA
        alpha, bias_vector = fit_bias_vector(model, bic_loader, len(label_to_idx), device,
                                             BIC_EPOCHS, BIC_LR, fit_alpha=fit_alpha)
        fold_bias_vector(model, bias_vector, alpha)
        # α is recorded either way — it is 1.0 in bias-only mode, which makes the stored
        # transform readable without having to know which mode wrote it.
        correction = {"mode": mode, "alpha": alpha, "bias": bias_vector, "n_old": n_old, "folded": True}
        summary = ("  " + (f"α={alpha:.4f}  " if fit_alpha else "")
                   + "b | " + _format_bias_vector(bias_vector, idx_to_label))
    else:
        alpha, beta = fit_bias_correction(model, bic_loader, n_old, device, BIC_EPOCHS, BIC_LR)
        fold_bias_correction(model, alpha, beta, n_old)
        correction = {"mode": BIC_MODE_SCALAR, "alpha": alpha, "beta": beta, "n_old": n_old, "folded": True}
        summary = f"  α={alpha:.4f}  β={beta:.4f}"

    after = evaluate(model, val_loader, device, n_old)

    logger.info("%s", summary)
    logger.info("  Val before | %.1f%% (new %s / old %s)", before["acc"], _pct(before["new_acc"]), _pct(before["old_acc"]))
    logger.info("  Val after  | %.1f%% (new %s / old %s)", after["acc"], _pct(after["new_acc"]), _pct(after["old_acc"]))

    save_checkpoint(
        checkpoint_path, model, label_to_idx, idx_to_label,
        whisper_model_name, freeze_encoder, after["acc"], extras["epoch"],
        exemplars=exemplars,
        bias_correction=correction,
        seed=extras["seed"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    data_dir = Path("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load BASE checkpoint
    logger.info("Loading checkpoint: %s", CHECKPOINT_PATH)
    old_model, label_to_idx, idx_to_label, whisper_model_name, base_freeze_encoder, _, _ = load_checkpoint(
        CHECKPOINT_PATH, str(device)
    )
    stored_exemplars = load_checkpoint_extras(CHECKPOINT_PATH)["exemplars"]
    old_labels = list(label_to_idx.keys())
    n_old = len(old_labels)
    logger.info("Existing classes (%d): %s", n_old, old_labels)

    # 2. Prompt for new command label and training config
    new_label = prompt_new_command()
    if new_label in label_to_idx:
        logger.error("Label '%s' already exists in this checkpoint. Aborting.", new_label)
        sys.exit(1)

    model_name, epochs, bic_mode = prompt_training_config()
    checkpoint_out = f"models/{date.today()}_{model_name}.pth"
    le_path = str(Path(checkpoint_out).with_name(f"{Path(checkpoint_out).stem}_LE.pth"))
    logger.info("Bias correction mode: %s", bic_mode)

    # 3. Generate data unless the label directory already exists
    label_dir = data_dir / new_label
    if label_dir.exists():
        logger.info("Data directory '%s' already exists, skipping generation.", label_dir)
    else:
        run_data_generation(new_label, data_dir)

    # 4. Freeze a copy of the pre-update head — this is the teacher. It consumes the same
    #    pooled features as the student, so no second encoder is needed.
    teacher_head = copy.deepcopy(old_model.classifier).to(device).eval()
    for p in teacher_head.parameters():
        p.requires_grad_(False)

    # 5. Expand label maps (new label appended at index N, old indices unchanged)
    label_to_idx[new_label] = n_old
    idx_to_label[n_old] = new_label
    logger.info("New label '%s' assigned index %d", new_label, n_old)

    # 6. Build expanded model
    logger.info("Building %d-class model (was %d)...", n_old + 1, n_old)
    model = expand_classifier(old_model, whisper_model_name, n_old + 1, device, head_dropout=HEAD_DROPOUT)
    del old_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 7. Resolve exemplars for the old classes
    logger.info("Resolving exemplars (%d per class)...", EXEMPLARS_PER_CLASS)
    try:
        exemplars = resolve_exemplars(
            stored_exemplars, old_labels, data_dir, model, EXEMPLARS_PER_CLASS, model.n_mels, device, BATCH_SIZE
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    # 8. Build the three disjoint splits
    new_files, _ = collect_files(new_label, data_dir)
    train_loader, bic_loader, val_loader = build_loaders(
        exemplars, new_label, new_files, data_dir, label_to_idx, model.n_mels, SEED
    )

    # 9. Stage 1: distillation + replay
    logger.info("Training '%s' for %d epochs with distillation + replay (CE weight %.2f) → %s",
                new_label, epochs, REPLAY_CE_WEIGHT, checkpoint_out)
    train_distillation(
        model, teacher_head, train_loader, val_loader, device,
        epochs, n_old, checkpoint_out, le_path,
        label_to_idx, idx_to_label, whisper_model_name, base_freeze_encoder,
        train_hidden_layer=TRAIN_HIDDEN_LAYER,
    )

    # 10. Select exemplars for the new class using the freshly trained model, so the next
    #     incremental step has them available.
    logger.info("Herding %d exemplars for '%s'...", EXEMPLARS_PER_CLASS, new_label)
    exemplars[new_label] = select_exemplars_herding(
        model, new_files, EXEMPLARS_PER_CLASS, model.n_mels, device, BATCH_SIZE
    )

    # 11. Stage 2: fit and fold the bias correction into both checkpoints
    for path in (checkpoint_out, le_path):
        apply_bias_correction(
            model, path, bic_loader, val_loader, device, n_old,
            label_to_idx, idx_to_label, whisper_model_name, base_freeze_encoder, exemplars,
            mode=bic_mode,
        )

    logger.info("Done. Checkpoints: %s and %s", checkpoint_out, le_path)


if __name__ == "__main__":
    main()
