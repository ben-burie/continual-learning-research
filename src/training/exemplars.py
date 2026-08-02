import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.dataset import CommandDataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_pooled_features(model, file_paths: list[str], n_mels: int, device, batch_size: int = 8) -> torch.Tensor:
    """Pooled encoder features [N, D] for file_paths, in order, un-augmented."""
    if not file_paths:
        return torch.empty(0)

    # Labels are irrelevant here — CommandDataset is reused purely for its preprocessing.
    ds = CommandDataset(file_paths, ["_"] * len(file_paths), {"_": 0}, n_mels, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    was_training = model.training
    model.eval()
    feats = []
    for mels, _, n_frames in loader:
        feats.append(model.pooled_features(mels.to(device), n_frames.to(device)).cpu())
    model.train(was_training)
    return torch.cat(feats)


def select_exemplars_herding(model, file_paths: list[str], k: int, n_mels: int, device,
                             batch_size: int = 8) -> list[str]:
    """iCaRL herding: greedily pick the k clips whose running feature mean best tracks
    the class mean.

    Returned most-representative-first, so callers can slice the head of the list when
    they need a subset.
    """
    if k <= 0 or not file_paths:
        return []
    if len(file_paths) <= k:
        return list(file_paths)

    feats = extract_pooled_features(model, file_paths, n_mels, device, batch_size)
    feats = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
    mu = feats.mean(dim=0)
    mu = mu / mu.norm().clamp(min=1e-8)

    selected: list[int] = []
    running = torch.zeros_like(mu)
    available = torch.ones(len(feats), dtype=torch.bool)

    for step in range(1, k + 1):
        # ‖mu − (Σ_selected + f_i) / step‖ for every candidate at once
        dist = (mu.unsqueeze(0) - (running.unsqueeze(0) + feats) / step).norm(dim=1)
        dist[~available] = float("inf")
        best = int(dist.argmin())
        selected.append(best)
        available[best] = False
        running = running + feats[best]

    return [file_paths[i] for i in selected]


def list_class_wavs(data_dir: Path, label: str) -> list[str]:
    class_dir = data_dir / label
    if not class_dir.exists():
        return []
    return sorted(str(f) for f in class_dir.iterdir() if f.suffix.lower() == ".wav")


def resolve_exemplars(stored: dict | None, labels: list[str], data_dir: Path, model, k: int,
                      n_mels: int, device, batch_size: int = 8) -> dict[str, list[str]]:
    """Return {label: [wav path, ...]} for each label.

    Reuses the checkpoint's stored list when every file is still on disk, otherwise
    herds a fresh set from data/<label>/. Raises if a label has neither.
    """
    stored = stored or {}
    resolved: dict[str, list[str]] = {}

    for label in labels:
        cached = stored.get(label)
        if cached and all(Path(p).exists() for p in cached):
            logger.info("  %s: reusing %d stored exemplars", label, len(cached))
            resolved[label] = list(cached)
            continue

        if cached:
            logger.warning("  %s: stored exemplar files are missing — re-selecting.", label)

        paths = list_class_wavs(data_dir, label)
        if not paths:
            raise FileNotFoundError(
                f"No exemplars stored for '{label}' and no .wav files under {data_dir / label}. "
                f"Regenerate the old-class data before running distillation training."
            )
        logger.info("  %s: herding %d exemplars from %d clips...", label, k, len(paths))
        resolved[label] = select_exemplars_herding(model, paths, k, n_mels, device, batch_size)

    return resolved


def partition_old_classes(exemplars: dict[str, list[str]], data_dir: Path, bic_val_per_class: int,
                          val_per_class: int) -> tuple[tuple[list, list], tuple[list, list], tuple[list, list]]:
    """Split the old classes three ways, with no clip appearing in more than one set.

    Returns (train, bic_val, val) as (paths, labels) pairs:
      train    — exemplars used for the distillation loss
      bic_val  — exemplars held out to fit the bias-correction parameters
      val      — non-exemplar clips, so retention is measured on genuinely unseen audio
    """
    train_paths, train_labels = [], []
    bic_paths, bic_labels = [], []
    val_paths, val_labels = [], []

    for label, paths in exemplars.items():
        n_train = max(len(paths) - bic_val_per_class, 1)
        # Herding order is most-representative-first, so the head of the list trains.
        ex_train, ex_bic = paths[:n_train], paths[n_train:]
        train_paths.extend(ex_train)
        train_labels.extend([label] * len(ex_train))
        bic_paths.extend(ex_bic)
        bic_labels.extend([label] * len(ex_bic))

        used = set(paths)
        held_out = [p for p in list_class_wavs(data_dir, label) if p not in used][:val_per_class]
        val_paths.extend(held_out)
        val_labels.extend([label] * len(held_out))

    return (train_paths, train_labels), (bic_paths, bic_labels), (val_paths, val_labels)
