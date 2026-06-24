import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.model.classifier import WhisperCommandClassifier
from src.training.dataset import build_dataloaders, load_data_from_dir
from src.training.trainer import train_model
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
CHECKPOINT = "models/BASE.pth"
WHISPER_MODEL = "turbo"
BATCH_SIZE = 8
LR = 1e-4
# Classification head: None = single linear layer; int = one hidden layer of this size
HEAD_HIDDEN_DIM = None
HEAD_DROPOUT = 0.0
SEED = 0

def select_commands(all_commands: list[str]) -> list[str]:
    print("\nAvailable commands:")
    for i, cmd in enumerate(all_commands, 1):
        print(f"  [{i}] {cmd}")

    print("\nEnter command numbers to include (e.g. 1 3 5), or press Enter to include all:")
    raw = input("  Selection: ").strip()

    if not raw:
        return all_commands

    selected = []
    for token in raw.split():
        if token.isdigit() and 1 <= int(token) <= len(all_commands):
            selected.append(all_commands[int(token) - 1])
        else:
            print(f"  Ignoring invalid selection: '{token}'")

    if not selected:
        print("No valid selections made.")
        sys.exit(1)

    return selected

def main():
    set_seed(SEED)
    all_data = load_data_from_dir(DATA_DIR)
    if not all_data:
        logger.error(f"No .wav files found in '{DATA_DIR}'")
        sys.exit(1)

    chosen = select_commands(sorted(all_data.keys()))
    data_dict = {k: all_data[k] for k in chosen}

    while True:
        try:
            epoch_input = int(input("\nNumber of epochs: ").strip())
            break
        except ValueError:
            print("Please enter a valid number.")

    unique_labels = sorted(data_dict.keys())
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    logger.info(f"Classes ({len(unique_labels)}): {unique_labels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = WhisperCommandClassifier(WHISPER_MODEL, len(unique_labels), freeze_encoder=True,
                                    head_hidden_dim=HEAD_HIDDEN_DIM, head_dropout=HEAD_DROPOUT)
    model.to(device)

    train_loader, val_loader = build_dataloaders(data_dict, label_to_idx, model.n_mels, BATCH_SIZE)
    logger.info(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")
    logger.info(f"Checkpoint: {CHECKPOINT}")

    train_model(
        model, train_loader, val_loader, device,
        epoch_input, LR, CHECKPOINT,
        label_to_idx, idx_to_label, WHISPER_MODEL, freeze_encoder=True,
        compute_fisher=True,
    )

if __name__ == "__main__":
    main()