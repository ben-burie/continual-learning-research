import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def icarl_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                            label_idxs: torch.Tensor, n_old: int, replay_ce_weight: float = 0.0,
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """iCaRL / LwF.MC loss: per-node binary cross-entropy over sigmoid outputs, plus an
    optional softmax cross-entropy replay term.

    Old nodes are supervised by the frozen pre-update network's soft targets, new nodes
    by the binary ground-truth label. Note that an old-class exemplar gets an all-zero
    target on the new nodes — its own label never appears as a hard target, so old-class
    supervision reaches the student only through the teacher. That is iCaRL as specified.

    On a single-Linear head over a frozen encoder that specification is degenerate. The
    per-node BCE decouples the output rows, and the old rows begin as an exact copy of the
    teacher over the exact same pooled features, so their target is already met: their
    gradient is *identically zero* on every batch (verifiable in float64 — in float32 the
    residual is round-off, which Adam's normalisation then inflates to a full-size step).
    Nothing the new class does can move them, and the exemplars only ever act as negatives
    for the new row.

    replay_ce_weight > 0 adds cross-entropy on the true labels. Its softmax normaliser
    couples every row, so the exemplars finally train the old rows and the distillation
    term becomes a genuine pull-back towards the teacher instead of a no-op. This is a
    deliberate departure from the paper — report such runs as replay + LwF, not iCaRL.

    The terms sit on different scales: distill and classification are summed over their
    nodes before the batch mean, while cross-entropy is a single mean, so a weight of 1.0
    already places replay well below distillation.

    Returns (total, distill_term, class_term, replay_term), each as it enters the total,
    so the three parts sum to it.
    """
    targets = torch.zeros_like(student_logits)
    targets[:, :n_old] = torch.sigmoid(teacher_logits.detach())

    is_new = label_idxs >= n_old
    targets[is_new, label_idxs[is_new]] = 1.0

    per_node = F.binary_cross_entropy_with_logits(student_logits, targets, reduction="none")
    distill = per_node[:, :n_old].sum(dim=1).mean()
    classification = per_node[:, n_old:].sum(dim=1).mean()

    # Applied to the whole batch, not just the exemplars: the new-class clips already carry
    # a hard target through `classification`, but keeping cross-entropy over every sample is
    # what makes this arm run the same objective as the CrossEntropyLoss baselines it is
    # being compared against.
    if replay_ce_weight:
        replay = replay_ce_weight * F.cross_entropy(student_logits, label_idxs)
    else:
        replay = student_logits.new_zeros(())

    return distill + classification + replay, distill, classification, replay


def fit_bias_correction(model, loader, n_old: int, device, epochs: int = 200,
                        lr: float = 1e-3) -> tuple[float, float]:
    """BiC stage 2: fit the two bias parameters on a small balanced validation set.

    Corrects only the new-class logits (q_k = α·o_k + β for k > n_old) under softmax
    cross-entropy, with the rest of the network frozen.
    """
    model.eval()

    # The network is frozen for this stage, so the logits never change — run the encoder
    # once and optimise the two scalars over the cached values.
    cached = []
    with torch.no_grad():
        for mels, label_idxs, n_frames in loader:
            cached.append((model(mels.to(device), n_frames.to(device)), label_idxs.to(device)))

    if not cached:
        logger.warning("Bias-correction set was empty — leaving logits uncorrected (α=1, β=0).")
        return 1.0, 0.0

    alpha = torch.ones(1, device=device, requires_grad=True)
    beta = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([alpha, beta], lr=lr)

    for epoch in range(epochs):
        total = 0.0
        for logits, label_idxs in cached:
            corrected = torch.cat([logits[:, :n_old], alpha * logits[:, n_old:] + beta], dim=1)
            loss = F.cross_entropy(corrected, label_idxs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if (epoch + 1) % 50 == 0:
            logger.info("  bias-correction epoch %3d/%d | loss %.4f | α=%.4f β=%.4f",
                        epoch + 1, epochs, total / len(cached), alpha.item(), beta.item())

    alpha, beta = float(alpha.item()), float(beta.item())
    if alpha <= 0:
        # Nothing in Eq. 4/5 constrains α to be positive, and on a very small validation
        # set it can fit to a sign flip, which inverts the new-class ranking.
        logger.warning("Bias correction fitted α=%.4f (≤ 0) — the validation set is likely too "
                       "small; consider raising BIC_VAL_PER_CLASS.", alpha)
    return alpha, beta


@torch.no_grad()
def fold_bias_correction(model, alpha: float, beta: float, n_old: int) -> None:
    """Absorb (α, β) into the new-class rows of the output layer.

    α·(w_k·f + b_k) + β == (α·w_k)·f + (α·b_k + β), so the corrected network is an
    ordinary classifier and nothing downstream (evaluate.py, load_checkpoint, the next
    incremental step) needs to know a correction was applied.
    """
    out = model.output_layer
    out.weight[n_old:] *= alpha
    out.bias[n_old:] = out.bias[n_old:] * alpha + beta
