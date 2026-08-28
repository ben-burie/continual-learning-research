import logging
import random
import sys
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.checkpoint import load_checkpoint, save_checkpoint
from src.model.classifier import WhisperCommandClassifier
from src.training.dataset import CommandDataset
from src.training.trainer import configure_head_training
from src.utils.seed import dataloader_generator, resolve_seed, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/BASE.pth"
N_SAMPLES = 150
BATCH_SIZE = 8
LR = 1e-3
HEAD_DROPOUT = None
SEED = resolve_seed()
OLD_VAL_SAMPLES_PER_CLASS = 15
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
            grad_update_prob = float(input("Gradient update probability for old head rows (0.0–1.0): "))
            if 0.0 <= grad_update_prob <= 1.0:
                break
            print("Must be between 0.0 and 1.0.")
        except ValueError:
            print("Please enter a valid number.")
    return model_name, epochs, grad_update_prob


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def run_data_generation(new_label: str, data_dir: Path) -> None:
    from scripts.generate_data import generate_audio_for_command
    from bark import preload_models
    logger.info("Loading Bark models...")
    preload_models()
    generate_audio_for_command(
        label=new_label,
        display_name=new_label.replace("_", " "),
        n_samples=N_SAMPLES,
        out_dir=data_dir,
    )


# ---------------------------------------------------------------------------
# Model expansion
# ---------------------------------------------------------------------------

def expand_classifier(old_model: WhisperCommandClassifier, whisper_model_name: str, n_total: int, device: torch.device,
                      head_dropout: float | None = None) -> WhisperCommandClassifier:
    # head_dropout=None inherits the base's rate; pass a float to override it for this continual run.
    dropout = old_model.head_dropout if head_dropout is None else head_dropout
    dropout = dropout or 0.0  # a base saved with head_dropout=None must not reach nn.Dropout(None)
    new_model = WhisperCommandClassifier(
        whisper_model_name, n_total, freeze_encoder=True,
        head_hidden_dim=old_model.head_hidden_dim, head_dropout=dropout,
    )
    new_model.encoder.load_state_dict(old_model.encoder.state_dict())

    old_out = old_model.output_layer
    new_out = new_model.output_layer
    n_old = old_out.out_features

    with torch.no_grad():
        # Copy the shared hidden layers verbatim (only the output layer grows with n_total).
        if isinstance(old_model.classifier, nn.Sequential):
            for new_layer, old_layer in zip(new_model.classifier[:-1], old_model.classifier[:-1]):
                if hasattr(old_layer, "weight"):
                    new_layer.load_state_dict(old_layer.state_dict())
        # Expand the output layer: copy old class rows, leave new rows freshly initialised.
        new_out.weight[:n_old].copy_(old_out.weight)
        new_out.bias[:n_old].copy_(old_out.bias)

    new_model.to(device)
    return new_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_files(new_label: str, data_dir: Path) -> tuple[list[str], list[str]]:
    new_dir = data_dir / new_label
    if not new_dir.exists():
        logger.error("New command data dir not found: %s", new_dir)
        sys.exit(1)
    new_files = [str(f) for f in new_dir.iterdir() if f.suffix.lower() == ".wav"]
    if not new_files:
        logger.error("No .wav files found in %s", new_dir)
        sys.exit(1)
    logger.info("New class '%s': %d files", new_label, len(new_files))
    return new_files, [new_label] * len(new_files)


def collect_old_val_files(label_to_idx: dict, new_label: str, data_dir: Path, per_class: int) -> tuple[list[str], list[str]]:
    # Sample a few clips from each *existing* class purely for validation, so val can catch
    # catastrophic forgetting. Classes whose data dir is missing are skipped (data may have
    # been generated in an earlier session and cleaned up) — the run still works without them.
    if per_class <= 0:
        return [], []
    paths, labels = [], []
    for label in label_to_idx:
        if label == new_label:
            continue
        class_dir = data_dir / label
        if not class_dir.exists():
            continue
        wavs = sorted(str(f) for f in class_dir.iterdir() if f.suffix.lower() == ".wav")
        if not wavs:
            continue
        sample = wavs[:per_class]
        paths.extend(sample)
        labels.extend([label] * len(sample))
    return paths, labels


def build_dataloaders(file_paths: list[str], labels: list[str], label_to_idx: dict, n_mels: int,
                      extra_val_paths: list[str] | None = None, extra_val_labels: list[str] | None = None,
                      seed: int = SEED) -> tuple[DataLoader, DataLoader]:
    # The split and the shuffle order both come from `seed`; they were pinned to 42 and
    # to the global RNG, which made a seed sweep re-run on identical data.
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    # Old-class clips only pad the validation set — training stays new-class-only by design.
    if extra_val_paths:
        val_paths = val_paths + extra_val_paths
        val_labels = val_labels + extra_val_labels
    train_ds = CommandDataset(train_paths, train_labels, label_to_idx, n_mels, augment=True)
    val_ds = CommandDataset(val_paths, val_labels, label_to_idx, n_mels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              generator=dataloader_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_continual(model: WhisperCommandClassifier, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
    epochs: int, n_old: int, checkpoint_path: str, label_to_idx: dict, idx_to_label: dict, whisper_model_name: str,
    grad_update_prob: float, freeze_encoder: bool = True,
    train_hidden_layer: bool = TRAIN_HIDDEN_LAYER) -> None:
    # By default only the output layer trains: the shared hidden layer (if any) stays
    # frozen so that updating on new-class-only data can't degrade previously learned
    # classes. train_hidden_layer=True lifts that, with nothing to guard against drift.
    out_layer = model.output_layer
    optimizer = torch.optim.AdamW(configure_head_training(model, train_hidden_layer), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    saved_any = False

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

            # Per batch, decide whether to zero old head row gradients.
            if random.random() >= grad_update_prob:
                with torch.no_grad():
                    if out_layer.weight.grad is not None:
                        out_layer.weight.grad[:n_old] = 0
                    if out_layer.bias.grad is not None:
                        out_layer.bias.grad[:n_old] = 0

            optimizer.step()

            t_loss += loss.item()
            t_correct += (logits.argmax(1) == label_idxs).sum().item()
            t_total += label_idxs.size(0)

        # --- Validate ---
        new_class_idx = n_old
        model.eval()
        v_loss = v_correct = v_total = 0
        v_new_correct = v_new_total = 0
        with torch.no_grad():
            for mels, label_idxs, n_frames in val_loader:
                mels = mels.to(device)
                label_idxs = label_idxs.to(device)
                n_frames = n_frames.to(device)
                logits = model(mels, n_frames)
                loss = criterion(logits, label_idxs)
                v_loss += loss.item()
                correct = logits.argmax(1) == label_idxs
                v_correct += correct.sum().item()
                v_total += label_idxs.size(0)
                new_mask = label_idxs == new_class_idx
                v_new_correct += correct[new_mask].sum().item()
                v_new_total += new_mask.sum().item()

        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        new_acc = 100 * v_new_correct / v_new_total if v_new_total else 0.0
        v_old_total = v_total - v_new_total
        old_acc = 100 * (v_correct - v_new_correct) / v_old_total if v_old_total else None
        old_str = f"{old_acc:.1f}%" if old_acc is not None else "n/a"
        logger.info(
            "Epoch %02d/%02d | Train %.4f / %.1f%% | Val %.4f / %.1f%% | new-cls %.1f%% | old-cls %s",
            epoch + 1, epochs,
            t_loss / len(train_loader), t_acc,
            v_loss / len(val_loader), v_acc,
            new_acc, old_str,
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint(
                checkpoint_path, model, label_to_idx, idx_to_label,
                whisper_model_name, freeze_encoder, v_acc, epoch + 1,
                seed=SEED,
            )
            saved_any = True
            logger.info("  → Best checkpoint saved (val_acc=%.1f%%)", v_acc)

        if epoch + 1 == epochs and not saved_any:
            save_checkpoint(
                checkpoint_path, model, label_to_idx, idx_to_label,
                whisper_model_name, freeze_encoder, v_acc, epoch + 1,
                seed=SEED,
            )
            saved_any = True
            logger.info("  → Final-epoch checkpoint saved as fallback (val_acc=%.1f%%)", v_acc)

    logger.info("Training complete. Best val accuracy: %.1f%%", best_val_acc)


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
    n_old = len(label_to_idx)
    logger.info("Existing classes (%d): %s", n_old, list(label_to_idx.keys()))

    # 2. Prompt for new command details + training config
    new_label = prompt_new_command()
    if new_label in label_to_idx:
        logger.error("Label '%s' already exists in this checkpoint. Aborting.", new_label)
        sys.exit(1)

    model_name, epochs, grad_update_prob = prompt_training_config()
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
    idx_to_label[new_idx] = new_label
    logger.info("New label '%s' assigned index %d", new_label, new_idx)

    # 5. Build expanded model
    logger.info("Building %d-class model (was %d)...", n_old + 1, n_old)
    model = expand_classifier(old_model, whisper_model_name, n_old + 1, device, head_dropout=HEAD_DROPOUT)
    del old_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 6. Collect training data and build loaders
    file_paths, labels = collect_files(new_label, data_dir)
    old_val_paths, old_val_labels = collect_old_val_files(
        label_to_idx, new_label, data_dir, OLD_VAL_SAMPLES_PER_CLASS
    )
    if old_val_paths:
        logger.info("Added %d old-class clips to validation (retention check).", len(old_val_paths))
    else:
        logger.warning("No old-class data found — validation measures the new class only.")
    train_loader, val_loader = build_dataloaders(
        file_paths, labels, label_to_idx, model.n_mels,
        extra_val_paths=old_val_paths, extra_val_labels=old_val_labels, seed=SEED,
    )
    logger.info("Train: %d  Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    # 7. Train
    logger.info(
        "Training new class '%s' for %d epochs → %s (grad_update_prob=%.2f)",
        new_label, epochs, checkpoint_out, grad_update_prob,
    )
    train_continual(
        model, train_loader, val_loader, device,
        epochs, n_old,
        checkpoint_out, label_to_idx, idx_to_label, whisper_model_name,
        grad_update_prob=grad_update_prob,
        freeze_encoder=base_freeze_encoder,
        train_hidden_layer=TRAIN_HIDDEN_LAYER,
    )

if __name__ == "__main__":
    main()