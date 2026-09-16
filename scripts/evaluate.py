import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.audio.preprocessing import preprocess_audio
from src.model.checkpoint import load_checkpoint, load_checkpoint_extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BATCH_SIZE = 32

def prompt_checkpoint() -> Path:
    models_dir = Path("models")
    options = sorted(models_dir.glob("*.pth")) if models_dir.exists() else []
    if options:
        print("\nAvailable checkpoints:")
        for p in options:
            print(f"  {p.name}")
    else:
        print("\nNo checkpoints found in models/")
    name = input("\nCheckpoint name (e.g. BASE.pth): ").strip()
    return models_dir / name

def prompt_test_dir() -> Path:
    name = input("Test data directory (e.g. test_data): ").strip()
    return Path(name)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate a checkpoint against a directory of labelled test audio. "
                    "Options omitted here are prompted for interactively.",
    )
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="Path to the checkpoint, including models/ (e.g. models/BASE.pth). "
                             "The interactive prompt asks for a bare name under models/; this "
                             "takes a full path so a sweep can point anywhere.")
    parser.add_argument("--test-dir", metavar="DIR",
                        help="Directory holding one <label>/ subdirectory of .wav files per "
                             "class (e.g. test_data_4). Its labels must match the checkpoint's "
                             "exactly.")
    parser.add_argument("--summary-csv", metavar="PATH",
                        help="Append one row for this model — overall and per-class accuracy — "
                             "to a shared CSV, creating it if needed. Use the same path across a "
                             "sweep to collect every arm in one table.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-file result table and print only the summary. "
                             "Per-file detail still goes to the CSV.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("checkpoint", "test_dir", "summary_csv"):
        value = getattr(args, name)
        if value is not None:
            value = value.strip()
            if not value:
                parser.error(f"--{name.replace('_', '-')} cannot be empty.")
            setattr(args, name, value)
    return args


def append_summary_row(summary_path: Path, checkpoint_path: Path, seed_tag: str, correct: int,
                       total: int, per_class_total: dict[str, int],
                       per_class_correct: dict[str, int]) -> None:
    row = {
        "model": checkpoint_path.stem,
        "seed": seed_tag,
        "overall_accuracy": f"{correct / total:.4f}" if total else "",
        "correct": correct,
        "total": total,
    }
    for label in sorted(per_class_total):
        n = per_class_total[label]
        row[f"acc_{label}"] = f"{per_class_correct[label] / n:.4f}" if n else ""
    fieldnames = list(row)

    existing: list[str] = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            existing = next(csv.reader(f), [])
    if existing and existing != fieldnames:
        logger.error("Summary CSV %s has columns %s, but this checkpoint needs %s — not "
                     "appending. Move the old file aside, or point --summary-csv elsewhere.",
                     summary_path, existing, fieldnames)
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Summary row appended -> %s", summary_path)


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

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else prompt_checkpoint()
    test_dir = Path(args.test_dir) if args.test_dir else prompt_test_dir()

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
    seed = load_checkpoint_extras(str(checkpoint_path))["seed"]
    logger.info("Checkpoint loaded (%d classes, seed=%s)", len(label_to_idx), seed)

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
    seed_tag = "unknown" if seed is None else str(seed)
    csv_path = csv_dir / f"{checkpoint_path.stem}_seed{seed_tag}_{timestamp}_evaluation.csv"

    rows = []
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    total = 0
    correct = 0

    if not args.quiet:
        print()
        print(f"{'FILE':<45} {'ACTUAL':<25} {'PREDICTED':<25} {'CONF':>6}  {'':>6}")
        print("-" * 115)

    for wav_path, actual_label, pred_label, conf in zip(
        all_wav_paths, all_actual_labels, all_pred_labels, all_confidences
    ):
        is_correct = pred_label == actual_label
        if not args.quiet:
            indicator = "[PASS]" if is_correct else "[FAIL]"
            print(f"{wav_path.name:<45} {actual_label:<25} {pred_label:<25} {conf:>6.1%}  {indicator}")

        rows.append({
            "seed": seed_tag,
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
    print(f"OVERALL ACCURACY: {correct}/{total}  ({overall_acc:.1%})   [seed {seed_tag}]")
    print()
    print(f"{'LABEL':<30} {'CORRECT':>8} {'TOTAL':>8} {'ACCURACY':>10} {'1 CLIP':>10}")
    print("-" * 72)
    for label in sorted(per_class_total):
        n = per_class_total[label]
        c = per_class_correct[label]
        print(f"{label:<30} {c:>8} {n:>8} {c/n:>10.1%} {1/n:>10.2%}")
    print("=" * 72)

    if args.summary_csv:
        append_summary_row(Path(args.summary_csv), checkpoint_path, seed_tag,
                           correct, total, per_class_total, per_class_correct)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "file", "actual_label", "predicted_label", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()