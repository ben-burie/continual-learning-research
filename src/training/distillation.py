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


@torch.no_grad()
def _cache_logits(model, loader, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(logits, label_idxs) per batch, computed once.

    Stage 2 freezes the network, so the logits cannot change while the calibration
    parameters are fit — re-running the encoder every epoch would cost orders of
    magnitude more than the fit itself.
    """
    model.eval()
    return [(model(mels.to(device), n_frames.to(device)), label_idxs.to(device))
            for mels, label_idxs, n_frames in loader]


def _warn_if_unconverged(label: str, initial: torch.Tensor, late: torch.Tensor,
                         final: torch.Tensor) -> None:
    """Flag a calibration that stopped because it ran out of epochs, not because it converged.

    Adam moves each parameter by at most ~lr per step, so the whole fit can travel no
    further than epochs x batches x lr — 0.6 at the committed defaults, over 20 clips in
    3 batches. That is ample when the head is nearly calibrated already and nowhere near
    enough when it is badly skewed, and the two cases look identical in the final numbers.
    A converged fit's parameters are barely moving by the end; one still travelling in its
    last tenth is reporting the budget rather than the data.
    """
    total = float((final - initial).norm())
    tail = float((final - late).norm())
    # A steady-rate fit puts a full tenth of its travel in its last tenth; 5% separates them.
    if total > 0 and tail > 0.05 * total:
        logger.warning("%s was still moving over the last 10%% of the fit (%.0f%% of its total "
                       "travel) — it is bounded by the step budget, not converged. Raise "
                       "BIC_EPOCHS or BIC_LR.", label, 100 * tail / total)


def fit_bias_correction(model, loader, n_old: int, device, epochs: int = 200,
                        lr: float = 1e-3) -> tuple[float, float]:
    """BiC stage 2: fit the two bias parameters on a small balanced validation set.

    Corrects only the new-class logits (q_k = α·o_k + β for k > n_old) under softmax
    cross-entropy, with the rest of the network frozen.

    Weight decay is off: AdamW's default would pull α and β towards 0 rather than towards
    the identity correction (α=1, β=0), which on ~20 fitting clips is a large and
    entirely unintended shrinkage.
    """
    cached = _cache_logits(model, loader, device)
    if not cached:
        logger.warning("Bias-correction set was empty — leaving logits uncorrected (α=1, β=0).")
        return 1.0, 0.0

    alpha = torch.ones(1, device=device, requires_grad=True)
    beta = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([alpha, beta], lr=lr, weight_decay=0.0)

    initial = torch.tensor([1.0, 0.0])
    late = initial.clone()

    for epoch in range(epochs):
        if epoch == int(0.9 * epochs):
            late = torch.tensor([alpha.item(), beta.item()])
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
    _warn_if_unconverged("α/β", initial, late, torch.tensor([alpha, beta]))
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


def fit_bias_vector(model, loader, n_classes: int, device, epochs: int = 200,
                    lr: float = 1e-3) -> list[float]:
    """Bias-only vector scaling: one additive offset per class, fit on the same balanced set.

    q = o + b with b in R^n_classes, under softmax cross-entropy and with the rest of the
    network frozen. Where fit_bias_correction rescales only the new-class logits and leaves
    the old ones untouched, this gives every class — old and new — its own offset, so the
    correction can also push down an old class the new one is being confused with. It buys
    that with strictly less power in another direction: nothing here rescales a logit, so
    the decision boundaries shift but never change orientation.

    Softmax is invariant to a constant added to every logit, so b is only identified up to
    that constant: the cross-entropy gradient w.r.t. b sums to zero across classes, and
    plain gradient descent from b=0 would therefore stay zero-sum. Adam's per-parameter
    normalisation breaks that, letting a meaningless common offset accumulate. The returned
    vector is re-centred to sum to zero — a no-op for this model's predictions and
    confidences, but it keeps the logged offsets comparable across runs, and stops an
    arbitrary shift from leaking into sigmoid(teacher_logits) if this checkpoint later
    becomes the teacher for another increment, where absolute logit level does matter.

    Weight decay is off for the same reason as in fit_bias_correction: b=0 is the identity
    correction, and shrinking towards it on ~20 clips is not a prior anyone chose.
    """
    cached = _cache_logits(model, loader, device)

    if not cached:
        logger.warning("Bias-correction set was empty — leaving logits uncorrected (b=0).")
        return [0.0] * n_classes

    n_logits = cached[0][0].shape[1]
    if n_logits != n_classes:
        raise ValueError(f"n_classes={n_classes} but the head emits {n_logits} logits")

    bias = torch.zeros(n_classes, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([bias], lr=lr, weight_decay=0.0)

    initial = torch.zeros(n_classes)
    late = initial.clone()

    for epoch in range(epochs):
        if epoch == int(0.9 * epochs):
            late = bias.detach().cpu().clone()
        total = 0.0
        for logits, label_idxs in cached:
            loss = F.cross_entropy(logits + bias, label_idxs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if (epoch + 1) % 50 == 0:
            # Centred, so these lines agree with the vector this eventually returns.
            shown = bias.detach() - bias.detach().mean()
            logger.info("  bias-vector epoch %3d/%d | loss %.4f | b=[%s]",
                        epoch + 1, epochs, total / len(cached),
                        ", ".join(f"{v:+.3f}" for v in shown.tolist()))

    _warn_if_unconverged("bias vector", initial, late, bias.detach().cpu())
    # Re-centring is a no-op for softmax, so it is applied after the convergence check
    # rather than before — the check should see the distance the fit actually travelled.
    centred = (bias.detach().cpu() - bias.detach().mean().cpu()).tolist()
    return [float(v) for v in centred]


@torch.no_grad()
def fold_bias_vector(model, bias_vector: list[float]) -> None:
    """Absorb the per-class offsets into the output layer's bias.

    (w_k·f + b_k) + v_k == w_k·f + (b_k + v_k), so — exactly as with fold_bias_correction —
    the corrected network stays an ordinary classifier and nothing downstream needs to know
    a correction was applied.
    """
    out = model.output_layer
    v = torch.as_tensor(bias_vector, dtype=out.bias.dtype, device=out.bias.device)
    if v.numel() != out.bias.numel():
        raise ValueError(f"bias vector has {v.numel()} entries but the head has {out.bias.numel()} classes")
    out.bias += v
