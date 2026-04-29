"""
Evaluate a trained checkpoint against a test directory (batched inference).

Usage:
    python scripts/evaluate.py --checkpoint models/voice_commander.pth
    python scripts/evaluate.py --checkpoint models/voice_commander.pth --test-dir test_data
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from src.audio.preprocessing import preprocess_audio
from src.model.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _scan_test_dir(test_dir: Path) -> dict[str, list[Path]]:
    """Return {label: [wav_path, ...]} for all subdirectories containing .wav files."""
    result = {}
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        wavs = sorted(subdir.glob("*.wav"))
        if wavs:
            result[subdir.name] = wavs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a voice command checkpoint against a test directory (batched).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pth model checkpoint to evaluate",
    )
    parser.add_argument(
        "--test-dir", default="test_data",
        help="Root test directory (one subfolder per command label)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading checkpoint: %s", checkpoint_path.name)
    model, label_to_idx, idx_to_label, _, _, _, _ = load_checkpoint(str(checkpoint_path), device=str(device))
    model.eval()
    n_mels = model.n_mels
    logger.info("Checkpoint loaded (%d classes)", len(label_to_idx))

    test_data = _scan_test_dir(test_dir)
    if not test_data:
        logger.error("No .wav files found under %s", test_dir)
        sys.exit(1)

    total_wav_count = sum(len(v) for v in test_data.values())
    logger.info("Found %d files across %d classes in %s", total_wav_count, len(test_data), test_dir)

    # Strict label match — fail loudly on any mismatch
    checkpoint_labels = set(label_to_idx.keys())
    test_labels = set(test_data.keys())
    if checkpoint_labels != test_labels:
        only_ckpt = checkpoint_labels - test_labels
        only_test = test_labels - checkpoint_labels
        if only_ckpt:
            logger.error("Labels in checkpoint but missing from test_data: %s", sorted(only_ckpt))
        if only_test:
            logger.error("Labels in test_data but missing from checkpoint: %s", sorted(only_test))
        sys.exit(1)

    # Preprocess all files upfront
    logger.info("Preprocessing %d audio files...", total_wav_count)
    all_wav_paths: list[Path] = []
    all_actual_labels: list[str] = []
    all_mels: list[torch.Tensor] = []
    all_n_frames: list[int] = []

    processed = 0
    for actual_label, wav_paths in sorted(test_data.items()):
        for wav_path in wav_paths:
            try:
                mel, n_frames = preprocess_audio(str(wav_path), n_mels=n_mels)
            except Exception as e:
                logger.warning("Skipping %s — preprocessing failed: %s", wav_path.name, e)
                continue
            all_wav_paths.append(wav_path)
            all_actual_labels.append(actual_label)
            all_mels.append(mel)
            all_n_frames.append(n_frames)
            processed += 1
            if processed % 50 == 0:
                logger.info("  Preprocessed %d / %d files...", processed, total_wav_count)

    logger.info("Preprocessing done: %d files", processed)

    # Batched inference
    n_batches = (len(all_mels) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("Running inference: %d files in %d batch(es) of up to %d", len(all_mels), n_batches, BATCH_SIZE)
    all_pred_labels: list[str] = []
    all_confidences: list[float] = []

    for batch_num, batch_start in enumerate(range(0, len(all_mels), BATCH_SIZE), start=1):
        batch_mels = torch.stack(all_mels[batch_start : batch_start + BATCH_SIZE]).to(device)
        batch_frames = torch.tensor(all_n_frames[batch_start : batch_start + BATCH_SIZE], device=device)
        logger.info("  Batch %d / %d (%d files)...", batch_num, n_batches, len(batch_mels))

        with torch.no_grad():
            logits = model(batch_mels, batch_frames)
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_idxs = probs.max(dim=-1)

        for pred_idx, conf in zip(pred_idxs.tolist(), confidences.tolist()):
            all_pred_labels.append(idx_to_label[pred_idx])
            all_confidences.append(conf)

    # CSV output
    csv_dir = Path("model_eval")
    csv_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = csv_dir / f"{checkpoint_path.stem}_{timestamp}_evaluation.csv"

    rows = []
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    total = 0
    correct = 0

    print()
    print(f"{'FILE':<45} {'ACTUAL':<25} {'PREDICTED':<25} {'CONF':>6}  {'':>6}")
    print("-" * 115)

    for wav_path, actual_label, pred_label, conf in zip(
        all_wav_paths, all_actual_labels, all_pred_labels, all_confidences
    ):
        is_correct = pred_label == actual_label
        indicator = "[PASS]" if is_correct else "[FAIL]"
        print(f"{wav_path.name:<45} {actual_label:<25} {pred_label:<25} {conf:>6.1%}  {indicator}")

        rows.append({
            "file": wav_path.name,
            "actual_label": actual_label,
            "predicted_label": pred_label,
            "confidence": f"{conf:.4f}",
            "correct": is_correct,
        })

        per_class_total[actual_label] += 1
        if is_correct:
            per_class_correct[actual_label] += 1
            correct += 1
        total += 1

    overall_acc = correct / total if total else 0.0
    print()
    print("=" * 60)
    print(f"OVERALL ACCURACY: {correct}/{total}  ({overall_acc:.1%})")
    print()
    print(f"{'LABEL':<30} {'CORRECT':>8} {'TOTAL':>8} {'ACCURACY':>10}")
    print("-" * 60)
    for label in sorted(per_class_total):
        n = per_class_total[label]
        c = per_class_correct[label]
        print(f"{label:<30} {c:>8} {n:>8} {c/n:>10.1%}")
    print("=" * 60)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "actual_label", "predicted_label", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()