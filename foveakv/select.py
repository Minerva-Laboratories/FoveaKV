"""Hot-set selection. Score tensors are (Hkv, G, E, T) logits (masked with -inf beyond qpos)."""
import numpy as np
from .attention import logprob


def always_hot(qpos, T, S=4, W=64):
    """(E, T) bool: sinks t < S, recent window qpos-W < t <= qpos."""
    t = np.arange(T)[None, :]
    qp = np.asarray(qpos)[:, None]
    return (t < S) | ((t > qp - W) & (t <= qp))


def k_for(qpos, frac, minimum=1):
    return np.maximum(minimum, np.ceil(frac * (np.asarray(qpos) + 1)).astype(int))


def sort_desc(score):
    return -np.sort(-score, axis=-1)


def topk_rows(score, k, srt=None):
    """score (..., E, T) with -inf invalid; k (E,) ints -> bool mask of the top-k per row.
    Pass srt=sort_desc(score) to reuse one sort across several k."""
    if srt is None:
        srt = sort_desc(score)
    kk = np.broadcast_to(np.minimum(np.maximum(k, 1), score.shape[-1]), srt.shape[:-1])   # per-row k (E,) or (...,E)
    thr = np.take_along_axis(srt, (kk - 1)[..., None], axis=-1)
    return (score >= thr) & (score > -np.inf)


def select(sel_scores, qpos, frac, S=4, W=64, mode="budget"):
    """sel_scores: (Hkv,G,E,T) estimated logits. Returns hot (Hkv,E,T) bool.
    mode='budget': rank by max over the group of estimated log-prob, keep k = frac*(p+1) tokens total
                   (always-hot tokens count inside the budget).
    mode='union' : per query head top-k (same k), then union over the group (size can reach G*k)."""
    Hkv, G, E, T = sel_scores.shape
    ah = always_hot(qpos, T, S, W)
    k = k_for(qpos, frac)
    lp = logprob(sel_scores)
    lp = np.where(ah[None, None], np.inf, lp)
    if mode == "budget":
        return topk_rows(lp.max(1), k)
    if mode == "union":
        return topk_rows(lp, k).any(1)
    raise ValueError(mode)


def pages(hot_tokens, page):
    """Expand a token-level hot mask to whole pages (any hot token -> page hot)."""
    if page == 1:
        return hot_tokens
    *lead, T = hot_tokens.shape
    n = -(-T // page)
    pad = np.zeros(tuple(lead) + (n * page - T,), bool)
    hp = np.concatenate([hot_tokens, pad], -1).reshape(*lead, n, page).any(-1)
    return np.repeat(hp, page, -1)[..., :T]


def select_pages(sel_scores, qpos, frac, page, S=4, W=64):
    """Page selection with a token budget: score page = max over its tokens (and group) of estimated
    log-prob; keep floor(k/page) pages; always-hot tokens are added on top at token granularity."""
    if page == 1:
        return select(sel_scores, qpos, frac, S, W)
    Hkv, G, E, T = sel_scores.shape
    lp = logprob(sel_scores).max(1)                                  # (Hkv,E,T)
    n = -(-T // page)
    pad = np.full((Hkv, E, n * page - T), -np.inf)
    ps = np.concatenate([lp, pad], -1).reshape(Hkv, E, n, page).max(-1)
    kp = np.maximum(1, k_for(qpos, frac) // page)
    hp = topk_rows(ps, kp)
    hot = np.repeat(hp, page, -1)[..., :T]
    valid = np.arange(T)[None, :] <= np.asarray(qpos)[:, None]
    return (hot & valid[None]) | always_hot(qpos, T, S, W)[None]
