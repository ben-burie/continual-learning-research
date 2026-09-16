import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = "scripts/distillation_continual_train.py"
EVAL_SCRIPT = "scripts/evaluate.py"

LABEL = "Open_Amazon"
EPOCHS = 20
TEST_DIR = "test_data_4"
# One appended row per evaluated model, collecting every arm in one table.
SUMMARY_CSV = "model_eval/master_summary.csv"
MODES = ["vector_alpha", "scalar"]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def launch(cmd: list[str], tag: str) -> int:
    """Run one subprocess, streaming its output. Returns the exit code."""
    print(f"\n{'=' * 70}\n[{tag}] {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)

    started = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - started

    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[{tag}] {status} in {elapsed / 60:.1f} min", flush=True)
    return result.returncode


def train(mode: str, seed: int) -> tuple[int, Path]:
    model_name = f"{mode}_s{seed}"
    expected = REPO_ROOT / "models" / f"{date.today()}_{model_name}.pth"
    code = launch([
        sys.executable, TRAIN_SCRIPT,
        "--label", LABEL,
        "--model-name", model_name,
        "--epochs", str(EPOCHS),
        "--bic-mode", mode,
        "--seed", str(seed),
    ], f"train {mode} / seed {seed}")
    return code, expected


def find_checkpoint(expected: Path, mode: str, seed: int) -> Path | None:
    if expected.exists():
        return expected
    candidates = sorted((REPO_ROOT / "models").glob(f"*_{mode}_s{seed}.pth"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def evaluate(checkpoint: Path, mode: str, seed: int) -> int:
    return launch([
        sys.executable, EVAL_SCRIPT,
        "--checkpoint", str(checkpoint.relative_to(REPO_ROOT)),
        "--test-dir", TEST_DIR,
        "--summary-csv", SUMMARY_CSV,
        "--quiet",
    ], f"eval {mode} / seed {seed}")


def main() -> None:
    failures = []
    started = time.time()

    for seed in seeds:
        for mode in MODES:
            code, expected = train(mode, seed)
            if code != 0:
                failures.append(f"{mode}_s{seed} (train)")
                continue

            checkpoint = find_checkpoint(expected, mode, seed)
            if checkpoint is None:
                print(f"[eval {mode} / seed {seed}] FAILED — no checkpoint at {expected}", flush=True)
                failures.append(f"{mode}_s{seed} (missing checkpoint)")
                continue

            if evaluate(checkpoint, mode, seed) != 0:
                failures.append(f"{mode}_s{seed} (eval)")

            print(f"{mode} COMPLETE.")

    total = len(seeds) * len(MODES)
    print(f"\n{'=' * 70}")
    print(f"{total - len(failures)}/{total} arms completed in {(time.time() - started) / 60:.1f} min")
    if failures:
        print("Failed: " + ", ".join(failures))
    print(f"Per-model summary: {SUMMARY_CSV}")
    print("Per-file CSVs:     model_eval/")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
