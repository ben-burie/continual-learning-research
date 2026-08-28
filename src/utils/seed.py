import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0
SEED_ENV_VAR = "ASR_SEED"


def resolve_seed(default: int = DEFAULT_SEED) -> int:
    """Seed for this run: $ASR_SEED when set, otherwise `default`.

    Read from the environment rather than hard-coded in each script so a sweep can vary
    it without editing source, and so the main.py menu path picks it up too — the
    training scripts parse no argv of their own.
    """
    raw = os.environ.get(SEED_ENV_VAR)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer — falling back to %d.", SEED_ENV_VAR, raw, default)
        return default


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d (deterministic=%s)", seed, deterministic)


def dataloader_generator(seed: int) -> torch.Generator:
    """A private RNG for a shuffling DataLoader.

    Without one the shuffle order is drawn from the global torch RNG, so it depends on how
    much randomness everything upstream happened to consume first — whisper.load_model
    alone draws a different amount for a different encoder size. An explicit generator
    makes batch order a function of the seed and nothing else.
    """
    return torch.Generator().manual_seed(seed)
