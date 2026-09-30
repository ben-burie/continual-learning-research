import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = "scripts/train.py"
EVAL_SCRIPT = "scripts/evaluate.py"

DATA_DIR = "data"
CHECKPOINT = "models/BASE.pth"
EPOCHS = 15
TEST_DIR = "test_data_3"
SUMMARY_CSV = "model_eval/base_seed_summary.csv"
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def launch(cmd: list[str], tag: str, stdin: str | None = None, env: dict | None = None) -> int:
    """Run one subprocess, streaming its output. Returns the exit code."""
    print(f"\n{'=' * 70}\n[{tag}] {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)

    started = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, input=stdin, text=True, env=env)
    elapsed = time.time() - started

    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[{tag}] {status} in {elapsed / 60:.1f} min", flush=True)
    return result.returncode


def command_selection() -> str:
    available = sorted(d.name for d in (REPO_ROOT / DATA_DIR).iterdir()
                       if d.is_dir() and any(f.suffix.lower() == ".wav" for f in d.iterdir()))
    wanted = sorted(d.name for d in (REPO_ROOT / TEST_DIR).iterdir() if d.is_dir())

    missing = [w for w in wanted if w not in available]
    if missing:
        sys.exit(f"Commands in {TEST_DIR} missing from {DATA_DIR}: {missing}")
    return " ".join(str(available.index(w) + 1) for w in wanted)


def train(seed: int, selection: str) -> int:
    # train.py takes no argv: the seed comes from $ASR_SEED, the rest from its prompts.
    env = {**os.environ, "ASR_SEED": str(seed)}
    return launch([sys.executable, TRAIN_SCRIPT], f"train base / seed {seed}",
                  stdin=f"{selection}\n{EPOCHS}\n", env=env)


def evaluate(seed: int) -> int:
    return launch([
        sys.executable, EVAL_SCRIPT,
        "--checkpoint", CHECKPOINT,
        "--test-dir", TEST_DIR,
        "--summary-csv", SUMMARY_CSV,
        "--quiet",
    ], f"eval base / seed {seed}")


def main() -> None:
    selection = command_selection()
    print(f"Training on {TEST_DIR} commands (menu selection: {selection})", flush=True)

    failures = []
    started = time.time()

    for seed in seeds:
        checkpoint = REPO_ROOT / CHECKPOINT
        # Remove the previous seed's checkpoint so a failed run can't be evaluated in its place.
        checkpoint.unlink(missing_ok=True)

        if train(seed, selection) != 0:
            failures.append(f"s{seed} (train)")
            continue

        if not checkpoint.exists():
            print(f"[eval base / seed {seed}] FAILED — no checkpoint at {checkpoint}", flush=True)
            failures.append(f"s{seed} (missing checkpoint)")
            continue

        if evaluate(seed) != 0:
            failures.append(f"s{seed} (eval)")

        print(f"Seed {seed} COMPLETE.")

    total = len(seeds)
    print(f"\n{'=' * 70}")
    print(f"{total - len(failures)}/{total} seeds completed in {(time.time() - started) / 60:.1f} min")
    if failures:
        print("Failed: " + ", ".join(failures))
    print(f"Per-model summary: {SUMMARY_CSV}")
    print("Per-file CSVs:     model_eval/")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
