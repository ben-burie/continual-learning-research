import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import signal
from scipy.io.wavfile import write as write_wav

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False

import torch

from bark import SAMPLE_RATE, generate_audio, preload_models

_torch_load = torch.load

def _torch_load_unsafe(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)

torch.load = _torch_load_unsafe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
N_SAMPLES = 150
SEED = 42

SPEAKERS = [f"v2/en_speaker_{i}" for i in range(10)]
NUM_IO_THREADS = 4

AUGMENTATION_POLICY = {  # 14 augs
    "noise": {"enabled": True, "snr_db": [35, 30, 25]},
    "tempo": {"enabled": _LIBROSA, "factors": [0.95, 1.05]},
    "pitch": {"enabled": _LIBROSA, "steps": [-1, 1]},
    "compression": {"enabled": True, "ratios": [2.0, 4.0]},
    "eq": {"enabled": True, "bands": ["low_cut", "mid_boost", "high_cut"]},
    "reverb": {"enabled": True, "strength": [0.3, 0.6]},
}


def _create_speech_variations(command: str) -> List[str]:
    words = command.split()
    variations = [
        command, command.lower(), command.upper(),
        f"Um, {command}", f"Uh, {command}", f"Hmm, {command}", f"Ah, {command}",
        f"{command}, please", f"{command} if you don't mind",
        f"Could you {command.lower()}",
        f"{command}!", f"Please {command}",
        f"{command}.", f"{command} now", f"{command} please",
    ]
    if len(words) > 2:
        mid = len(words) // 2
        variations.append(f"{' '.join(words[:mid])}... {' '.join(words[mid:])}")
        variations.append(f"{words[0]}... {' '.join(words[1:])}")

    seen: set = set()
    unique = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _apply_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    power = np.mean(audio ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    return audio + np.random.normal(0, np.sqrt(noise_power), len(audio))


def _apply_compression(audio: np.ndarray, ratio: float, threshold_db: float = -20) -> np.ndarray:
    threshold = 10 ** (threshold_db / 20)
    out = np.copy(audio)
    mask = np.abs(audio) > threshold
    out[mask] = np.sign(audio[mask]) * threshold + (np.abs(audio[mask]) - threshold) / ratio
    return out


def _apply_eq(audio: np.ndarray, band: str) -> np.ndarray:
    nyquist = SAMPLE_RATE / 2
    if band == "low_cut":
        sos = signal.butter(4, 300 / nyquist, btype="high", output="sos")
    elif band == "mid_boost":
        sos = signal.butter(4, [800 / nyquist, 3000 / nyquist], btype="band", output="sos")
    elif band == "high_cut":
        sos = signal.butter(4, 7000 / nyquist, btype="low", output="sos")
    else:
        return audio
    return signal.sosfilt(sos, audio)


def _apply_reverb(audio: np.ndarray, strength: float) -> np.ndarray:
    base_delay = int(0.05 * SAMPLE_RATE)
    delays = [base_delay, int(base_delay * 1.5), int(base_delay * 2.0)]
    decays = [strength * 0.5, strength * 0.3, strength * 0.15]
    reverb = np.zeros_like(audio)
    for delay, decay in zip(delays, decays):
        delayed = np.zeros_like(audio)
        delayed[delay:] = audio[:-delay] * decay
        reverb += delayed
    return audio + reverb


def _apply_augmentations(audio: np.ndarray, policy: dict) -> List[np.ndarray]:
    versions = [audio]

    if policy["noise"]["enabled"]:
        for snr in policy["noise"]["snr_db"]:
            try:
                versions.append(_apply_noise(audio, snr))
            except Exception as e:
                logger.warning("Noise aug failed (snr=%s): %s", snr, e)

    if _LIBROSA and policy["tempo"]["enabled"]:
        for factor in policy["tempo"]["factors"]:
            try:
                versions.append(librosa.effects.time_stretch(audio, rate=factor))
            except Exception as e:
                logger.warning("Tempo aug failed (factor=%s): %s", factor, e)

    if _LIBROSA and policy["pitch"]["enabled"]:
        for steps in policy["pitch"]["steps"]:
            try:
                versions.append(librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps))
            except Exception as e:
                logger.warning("Pitch aug failed (steps=%s): %s", steps, e)

    if policy["compression"]["enabled"]:
        for ratio in policy["compression"]["ratios"]:
            try:
                versions.append(_apply_compression(audio, ratio))
            except Exception as e:
                logger.warning("Compression aug failed (ratio=%s): %s", ratio, e)

    if policy["eq"]["enabled"]:
        for band in policy["eq"]["bands"]:
            try:
                versions.append(_apply_eq(audio, band))
            except Exception as e:
                logger.warning("EQ aug failed (band=%s): %s", band, e)

    if policy["reverb"]["enabled"]:
        for strength in policy["reverb"]["strength"]:
            try:
                versions.append(_apply_reverb(audio, strength))
            except Exception as e:
                logger.warning("Reverb aug failed (strength=%s): %s", strength, e)

    return versions


def _normalize(audio: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio * (10 ** (target_lufs / 20) / rms)
    return np.tanh(audio)


def _save_batch(batch: List[Tuple[np.ndarray, Path]]) -> None:
    def _save(audio: np.ndarray, path: Path) -> None:
        write_wav(str(path), SAMPLE_RATE, audio)

    with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as pool:
        futures = [pool.submit(_save, audio, path) for audio, path in batch]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error("Failed to save audio file: %s", e)


def generate_audio_for_command(label: str, display_name: str, n_samples: int, out_dir: Path, seed: int = SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)

    out_path = out_dir / label
    out_path.mkdir(parents=True, exist_ok=True)

    variations = _create_speech_variations(display_name)
    logger.info("Generating '%s': %d text variations, target %d samples", label, len(variations), n_samples)

    count = 0
    generation_idx = 0
    while count < n_samples:
        speaker_idx = generation_idx % len(SPEAKERS)
        speaker = SPEAKERS[speaker_idx]
        text = variations[generation_idx % len(variations)]
        generation_idx += 1

        try:
            raw = generate_audio(text, history_prompt=speaker)
            raw = _normalize(raw)
            augmented = _apply_augmentations(raw, AUGMENTATION_POLICY)
        except Exception as e:
            logger.warning("Bark generation failed (speaker=%s, text=%r): %s", speaker, text, e)
            continue

        batch = []
        for aug in augmented:
            if count >= n_samples:
                break
            audio = np.array(aug, dtype=np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * (0.95 / max_val)
            batch.append((audio, out_path / f"{label}_speaker{speaker_idx:02d}_aug{count:04d}.wav"))
            count += 1

        _save_batch(batch)

    logger.info("'%s': wrote %d files to %s", label, count, out_path)
    return count


def main() -> None:
    print("Enter commands one per line. Leave blank and press Enter when done.")
    commands = []
    while True:
        line = input(f"  Command {len(commands) + 1}: ").strip()
        if not line:
            break
        commands.append(line)

    if not commands:
        print("No commands entered.")
        sys.exit(1)

    logger.info("Loading Bark AI models...")
    preload_models()
    logger.info("Bark models loaded.")

    results = {}
    for command in commands:
        label = command.replace(" ", "_")
        results[label] = generate_audio_for_command(label=label, display_name=command, n_samples=N_SAMPLES, out_dir=DATA_DIR)

    print("\n" + "=" * 50)
    for label, count in results.items():
        print(f"  {label}: {count} files -> {(DATA_DIR / label).resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()