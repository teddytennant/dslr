import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass
class DSLRResult:
    reconstruction: np.ndarray
    stored_rows: np.ndarray
    rank: int
    cost: int
    error: float
    floor: float
    qmax: float
    lam: float
    tau2: float
    S: np.ndarray
    L: np.ndarray
    split_floor: float | None = None
    k: int = 0


@dataclass
class ShapeFit:
    reconstruction: np.ndarray
    S: np.ndarray
    L: np.ndarray
    error: float
    stored_rows: np.ndarray
    rank: int
    cost: int


def cost(n, m, rank, k):
    n = int(n)
    m = int(m)
    rank = int(rank)
    k = int(k)
    if k >= n:
        return n * m
    return k * m + rank * (n - k + m)


def _as_matrix(W):
    arr = jnp.asarray(W)
    if arr.ndim != 2:
        raise ValueError("W must be a 2-dimensional array of shape (n, m)")
    arr = arr.astype(jnp.float64)
    n, m = int(arr.shape[0]), int(arr.shape[1])
    if n < 1 or m < 1:
        raise ValueError("W must have at least one row and one column")
    return np.asarray(arr)


def _svd(block):
    u, s, vh = jnp.linalg.svd(jnp.asarray(block), full_matrices=False)
    return np.asarray(u), np.asarray(s), np.asarray(vh)


def _tails_from_s2(s2, rmax):
    T = np.zeros(rmax + 1, dtype=np.float64)
    if s2.size:
        rev = np.cumsum(np.asarray(s2, dtype=np.float64)[::-1])[::-1]
        ncopy = min(int(s2.size), rmax)
        T[:ncopy] = rev[:ncopy]
    return T


def _row_prefix(row_energy):
    R = np.zeros(row_energy.size + 1, dtype=np.float64)
    if row_energy.size:
        R[1:] = np.cumsum(np.sort(row_energy)[::-1])
    return R


def _floor_at(T, R, rank, k):
    rmax = T.size - 1
    spectral = 0.0 if rank + k >= rmax else float(T[rank + k])
    tr = 0.0 if rank >= T.size else float(T[rank])
    if k >= R.size:
        rk = float(R[-1])
    elif k < 0:
        rk = 0.0
    else:
        rk = float(R[k])
    return max(spectral, tr - rk, 0.0)


def _floor_grid(T, R):
    rmax = T.size - 1
    n = R.size - 1
    rs = np.arange(rmax + 1)[:, None]
    ks = np.arange(n + 1)[None, :]
    spectral = T[np.minimum(rs + ks, rmax)]
    row_floor = T[rs] - R[ks]
    return np.maximum(np.maximum(spectral, row_floor), 0.0)


def universal_floor(W, rank, k):
    W = _as_matrix(W)
    n, m = W.shape
    rank = int(rank)
    k = int(k)
    if rank < 0 or k < 0:
        raise ValueError("rank and k must be non-negative")
    if k > n:
        raise ValueError("k cannot exceed the number of rows")
    _, s, _ = _svd(W)
    T = _tails_from_s2(s ** 2, min(n, m))
    R = _row_prefix(np.sum(W ** 2, axis=1))
    return _floor_at(T, R, rank, k)


def mp_lower_half_factor(gamma, n_grid=4096):
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        return 1.0
    gamma = min(gamma, 1.0)
    sqrt_g = math.sqrt(gamma)
    a = (1.0 - sqrt_g) ** 2
    b = (1.0 + sqrt_g) ** 2
    width = b - a
    if width <= 0.0:
        return 1.0
    theta = np.linspace(0.0, 0.5 * math.pi, n_grid)
    theta[0] = 1e-12
    theta[-1] = 0.5 * math.pi - 1e-12
    s = np.sin(theta)
    c = np.cos(theta)
    x = a + width * (s * s)
    w = (width * width) * (s * s) * (c * c) / (math.pi * gamma * x)
    dth = float(theta[1] - theta[0])
    trap = np.ones(n_grid)
    trap[0] = 0.5
    trap[-1] = 0.5
    mass = w * trap * dth
    total = float(mass.sum())
    if not math.isfinite(total) or total <= 0.0:
        return 1.0
    mass = mass / total
    cdf = np.cumsum(mass)
    xmass = x * mass
    idx = int(np.searchsorted(cdf, 0.5))
    idx = min(max(idx, 1), n_grid - 1)
    prev = float(cdf[idx - 1])
    span = float(cdf[idx] - prev)
    frac = 0.0 if span <= 0.0 else (0.5 - prev) / span
    frac = min(max(frac, 0.0), 1.0)
    lower = float(xmass[:idx].sum() + frac * xmass[idx])
    factor = lower / 0.5
    if not math.isfinite(factor) or factor <= 0.0:
        return 1.0
    return float(factor)


def _residual_variance(W, T):
    n, m = W.shape
    rmax = min(n, m)
    var = float(T[rmax // 2]) / (n * m)
    if not math.isfinite(var) or var <= 0.0:
        var = float(np.mean(W ** 2))
    if not math.isfinite(var) or var <= 0.0:
        var = 1e-12
    return var


def _tau2(s2, n_block, m, T, W):
    p = min(int(n_block), int(m))
    larger = max(int(n_block), int(m))
    if p < 2 or s2.size < 2:
        return _residual_variance(W, T)
    half = int(s2.size) // 2
    if half < 1 or larger < 1:
        return _residual_variance(W, T)
    smallest = np.partition(np.asarray(s2, dtype=np.float64), half - 1)[:half]
    raw = float(np.mean(smallest)) / larger
    factor = mp_lower_half_factor(p / larger)
    tau2 = raw / factor
    if not math.isfinite(tau2) or tau2 <= 0.0:
        tau2 = _residual_variance(W, T)
    if not math.isfinite(tau2) or tau2 <= 0.0:
        tau2 = 1e-12
    return float(tau2)


def _ebic_lambda(tau2, n, m):
    n_shapes = (min(n, m) + 1) * (n + 1)
    return float(tau2) * (math.log(n * m) + 2.0 * math.log(n_shapes))


def _flag_rows(W, vh, row_energy):
    n, m = W.shape
    rprobe = min(15, min(n, m) - 1)
    if n == 1 or rprobe < 1:
        return np.zeros(n, dtype=bool)
    kflag = n // 2
    if kflag < 1:
        return np.zeros(n, dtype=bool)
    captured = np.sum((W @ vh[:rprobe].T) ** 2, axis=1)
    residual = np.maximum(row_energy - captured, 0.0)
    flagged = np.zeros(n, dtype=bool)
    flagged[np.argsort(-residual, kind="mergesort")[:kflag]] = True
    return flagged


def _surrogate_grid(W, vh_hat, row_energy, T):
    n, m = W.shape
    rmax = min(n, m)
    nsv = 0 if vh_hat.size == 0 else vh_hat.shape[0]
    rho = np.broadcast_to(row_energy, (rmax + 1, n)).copy()
    if nsv > 0:
        captured = np.cumsum((W @ vh_hat.T) ** 2, axis=1)
        for r in range(1, rmax + 1):
            rho[r] = row_energy - captured[:, min(r, nsv) - 1]
    rho = np.maximum(rho, 0.0)
    E_hat = np.zeros((rmax + 1, n + 1), dtype=np.float64)
    orders = np.empty((rmax + 1, n), dtype=np.int64)
    for r in range(rmax + 1):
        order = np.argsort(-rho[r], kind="mergesort")
        orders[r] = order
        sorted_rho = rho[r, order]
        if n:
            E_hat[r, :n] = np.cumsum(sorted_rho[::-1])[::-1]
        E_hat[r, n] = 0.0
        E_hat[r, 0] = T[r]
    return E_hat, orders


def _cost_grid(n, m):
    rmax = min(n, m)
    rs = np.arange(rmax + 1, dtype=np.float64)[:, None]
    ks = np.arange(n + 1, dtype=np.float64)[None, :]
    C = ks * m + rs * (n - ks + m)
    C = np.array(C, copy=True)
    C[:, n] = float(n * m)
    return C


def _pick(E_hat, C, mode, budget, error_target, lam):
    r_idx, k_idx = np.indices(E_hat.shape)
    if mode == "ebic":
        score = E_hat + float(lam) * C
        order = np.lexsort((k_idx.ravel(), r_idx.ravel(), C.ravel(), score.ravel()))
    elif mode == "budget":
        if budget is None:
            feasible = np.ones(E_hat.shape, dtype=bool)
        else:
            feasible = C <= float(budget)
        if not np.any(feasible):
            return 0, 0
        score = np.where(feasible, E_hat, np.inf)
        order = np.lexsort((k_idx.ravel(), r_idx.ravel(), C.ravel(), score.ravel()))
    elif mode == "error_target":
        target = 0.0 if error_target is None else float(error_target)
        feasible = E_hat <= target
        if not np.any(feasible):
            order = np.lexsort((k_idx.ravel(), r_idx.ravel(), C.ravel(), E_hat.ravel()))
        else:
            score = np.where(feasible, C, np.inf)
            order = np.lexsort((k_idx.ravel(), r_idx.ravel(), E_hat.ravel(), score.ravel()))
    else:
        raise ValueError("mode must be 'ebic', 'budget', or 'error_target'")
    rank, k = np.unravel_index(int(order[0]), E_hat.shape)
    return int(rank), int(k)


def _reconstruct(W, rank, stored):
    n, m = W.shape
    stored = np.asarray(stored, dtype=np.int64)
    S = np.zeros((n, m), dtype=np.float64)
    L = np.zeros((n, m), dtype=np.float64)
    if stored.size:
        S[stored] = W[stored]
    mask = np.ones(n, dtype=bool)
    if stored.size:
        mask[stored] = False
    unstored = np.flatnonzero(mask)
    if rank > 0 and unstored.size:
        u, s, vh = _svd(W[unstored])
        rr = min(int(rank), int(s.size))
        if rr > 0:
            L[unstored] = (u[:, :rr] * s[:rr]) @ vh[:rr]
    return S, L, S + L


def fit_shape(W, rank, stored_rows=()):
    W = _as_matrix(W)
    n, m = W.shape
    rank = int(rank)
    if rank < 0:
        raise ValueError("rank must be non-negative")
    stored = np.unique(np.asarray(list(stored_rows), dtype=np.int64))
    if stored.size and (stored[0] < 0 or stored[-1] >= n):
        raise ValueError("stored row index out of range")
    S, L, recon = _reconstruct(W, rank, stored)
    err = float(np.sum((W - recon) ** 2))
    return ShapeFit(
        reconstruction=recon,
        S=S,
        L=L,
        error=err,
        stored_rows=stored,
        rank=rank,
        cost=cost(n, m, rank, int(stored.size)),
    )


def _partition_counts(n, m):
    # 0.8 = 4/5. Integer division is floor(n / (0.8 m)) without binary rounding.
    t = (5 * n) // (4 * m)
    t = max(1, min(t, n))
    counts = []
    for _ in range(5):
        if t >= 2:
            counts.append(t)
        nxt = t // 2
        if nxt < 2:
            break
        t = nxt
    return counts


def _block_tables(block):
    row_energy = np.sum(block ** 2, axis=1)
    _, s, _ = _svd(block)
    rmax = min(block.shape[0], block.shape[1])
    return _tails_from_s2(s ** 2, rmax), _row_prefix(row_energy)


def split_floor(W, rank, k, universal=None):
    W = _as_matrix(W)
    n, m = W.shape
    rank = int(rank)
    k = int(k)
    if universal is None:
        universal = universal_floor(W, rank, k)
    universal = float(universal)
    best = universal
    row_energy = np.sum(W ** 2, axis=1)
    order = np.argsort(-row_energy, kind="mergesort")
    for t in _partition_counts(n, m):
        blocks = [order[i::t] for i in range(t)]
        base = 0.0
        deltas = []
        for idx in blocks:
            T_b, R_b = _block_tables(W[idx])
            prev = _floor_at(T_b, R_b, rank, 0)
            base += prev
            for i in range(1, idx.size + 1):
                nxt = _floor_at(T_b, R_b, rank, i)
                deltas.append(min(nxt - prev, 0.0))
                prev = nxt
        value = base
        if k > 0 and deltas:
            deltas.sort()
            value = base + float(np.sum(deltas[:k]))
        if value > best:
            best = value
    if not math.isfinite(best):
        return None
    return float(best)


def decompose(W, mode="ebic", budget=None, error_target=None, lam=None, shape=None):
    W = _as_matrix(W)
    n, m = W.shape
    rmax = min(n, m)
    if mode not in ("ebic", "budget", "error_target"):
        raise ValueError("mode must be 'ebic', 'budget', or 'error_target'")

    _, s, vh = _svd(W)
    s2 = s ** 2
    T = _tails_from_s2(s2, rmax)
    row_energy = np.sum(W ** 2, axis=1)
    R = _row_prefix(row_energy)

    flagged = _flag_rows(W, vh, row_energy)
    unflagged = W[~flagged]
    if unflagged.shape[0] == 0:
        vh_hat = np.zeros((0, m), dtype=np.float64)
        s2_hat = np.zeros(0, dtype=np.float64)
    else:
        _, s_hat, vh_hat = _svd(unflagged)
        s2_hat = s_hat ** 2
    tau2 = _tau2(s2_hat, unflagged.shape[0], m, T, W)
    if lam is None:
        lam_used = _ebic_lambda(tau2, n, m)
    else:
        lam_used = float(lam)

    E_hat, orders = _surrogate_grid(W, vh_hat, row_energy, T)
    C = _cost_grid(n, m)
    if shape is None:
        rank, k = _pick(E_hat, C, mode, budget, error_target, lam_used)
    else:
        rank, k = int(shape[0]), int(shape[1])
        if rank < 0 or rank > rmax or k < 0 or k > n:
            raise ValueError("shape (rank, k) is outside the grid")
    if k == 0:
        stored = np.zeros(0, dtype=np.int64)
    else:
        stored = np.sort(orders[rank, :k])

    S, L, recon = _reconstruct(W, rank, stored)
    err = float(np.sum((W - recon) ** 2))
    floors = _floor_grid(T, R)
    floor = float(floors[rank, k])
    chosen_cost = cost(n, m, rank, k)
    q_hat = err + lam_used * chosen_cost
    qmax = float(q_hat - np.min(floors + lam_used * C))
    tightened = split_floor(W, rank, k, universal=floor)
    return DSLRResult(
        reconstruction=recon,
        stored_rows=stored,
        rank=rank,
        cost=chosen_cost,
        error=err,
        floor=floor,
        qmax=qmax,
        lam=lam_used,
        tau2=tau2,
        S=S,
        L=L,
        split_floor=tightened,
        k=k,
    )
