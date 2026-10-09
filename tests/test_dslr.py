import math

import numpy as np

import dslr
from dslr.decompose import cost, fit_shape, split_floor, universal_floor


def _tails(W):
    n, m = W.shape
    s2 = np.linalg.svd(W, compute_uv=False) ** 2
    rmax = min(n, m)
    T = np.zeros(rmax + 1)
    T[:rmax] = np.cumsum(s2[::-1])[::-1]
    return T


def _row_prefix(W):
    row_energy = np.sum(W ** 2, axis=1)
    R = np.zeros(W.shape[0] + 1)
    R[1:] = np.cumsum(np.sort(row_energy)[::-1])
    return row_energy, R


def _paper_floor(W, rank, k):
    T = _tails(W)
    _, R = _row_prefix(W)
    rmax = min(W.shape)
    spectral = 0.0 if rank + k >= rmax else float(T[rank + k])
    return max(spectral, float(T[rank]) - float(R[k]), 0.0)


def _assert_disjoint(W, res):
    n = W.shape[0]
    stored = np.asarray(res.stored_rows)
    unstored = np.setdiff1d(np.arange(n), stored)
    np.testing.assert_array_equal(res.reconstruction[stored], W[stored])
    assert np.all(res.S[unstored] == 0)
    assert np.all(res.L[stored] == 0)
    s_nz = np.any(res.S != 0, axis=1)
    l_nz = np.any(res.L != 0, axis=1)
    assert not np.any(s_nz & l_nz)
    assert np.linalg.matrix_rank(res.L) <= res.rank
    np.testing.assert_allclose(res.reconstruction, res.S + res.L)


def test_k0_matches_truncated_svd():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(12, 7))
    u, s, vh = np.linalg.svd(W, full_matrices=False)
    for rank in (0, 1, 3, min(W.shape)):
        res = dslr.decompose(W, shape=(rank, 0))
        trunc = np.zeros_like(W) if rank == 0 else (u[:, :rank] * s[:rank]) @ vh[:rank]
        np.testing.assert_allclose(res.reconstruction, trunc, rtol=1e-8, atol=1e-8)
        assert res.stored_rows.size == 0
        assert res.rank == rank
        assert res.k == 0
        assert np.all(res.S == 0)
        fit = fit_shape(W, rank, [])
        np.testing.assert_allclose(fit.reconstruction, trunc, rtol=1e-8, atol=1e-8)


def test_disjointness():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(18, 9))
    forced = dslr.decompose(W, shape=(2, 4))
    assert forced.k == 4
    assert forced.stored_rows.size == 4
    _assert_disjoint(W, forced)
    _assert_disjoint(W, dslr.decompose(W, mode="ebic"))


def test_cost_formula():
    n, m = 11, 7
    for rank in range(0, min(n, m) + 1):
        for k in range(0, n + 1):
            c = cost(n, m, rank, k)
            if k == n:
                assert c == n * m
            else:
                assert c == k * m + rank * (n - k + m)
            if rank > 0 and 0 < k < n:
                assert c < rank * (n + m) + k * m


def test_universal_floor_bounds_disjoint_error():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(15, 8))
    n, m = W.shape
    _, s, vh = np.linalg.svd(W, full_matrices=False)
    row_energy, _ = _row_prefix(W)
    shapes = [
        (0, 0),
        (0, 3),
        (1, 0),
        (1, 2),
        (2, 4),
        (3, 1),
        (min(n, m), 0),
        (2, n),
        (0, n),
        (4, 5),
    ]
    for rank, k in shapes:
        floor = _paper_floor(W, rank, k)
        assert abs(universal_floor(W, rank, k) - floor) <= 1e-8
        if rank == 0:
            rho = row_energy
        else:
            rho = row_energy - np.sum((W @ vh[:rank].T) ** 2, axis=1)
        stored = np.argsort(-rho, kind="mergesort")[:k]
        fit = fit_shape(W, rank, stored)
        assert fit.error >= floor - 1e-6


def test_certificate():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(16, 10))
    res = dslr.decompose(W, mode="ebic")
    n, m = W.shape
    rank, k = res.rank, res.k
    expected_floor = _paper_floor(W, rank, k)
    assert abs(res.floor - expected_floor) <= 1e-6
    assert res.error >= res.floor - 1e-5
    assert res.qmax >= -1e-6

    T = _tails(W)
    _, R = _row_prefix(W)
    rmax = min(n, m)
    rs = np.arange(rmax + 1)[:, None]
    ks = np.arange(n + 1)[None, :]
    spectral = T[np.minimum(rs + ks, rmax)]
    floors = np.maximum(np.maximum(spectral, T[rs] - R[ks]), 0.0)
    C = ks * m + rs * (n - ks + m)
    C = np.array(C, copy=True)
    C[:, n] = n * m
    qmax = res.error + res.lam * res.cost - np.min(floors + res.lam * C)
    assert abs(res.qmax - qmax) <= 1e-6
    assert res.cost == cost(n, m, rank, k)

    if res.split_floor is not None:
        assert res.split_floor >= res.floor - 1e-8
        assert res.split_floor <= res.error + 1e-5


def test_split_floor_bound():
    rng = np.random.default_rng(7)
    W = rng.normal(size=(20, 8))
    n, m = W.shape
    row_energy = np.sum(W ** 2, axis=1)
    _, _, vh = np.linalg.svd(W, full_matrices=False)
    for rank, k in ((0, 2), (1, 3), (2, 4), (3, 1), (1, 0), (0, n)):
        uni = universal_floor(W, rank, k)
        tightened = split_floor(W, rank, k)
        assert tightened is not None
        assert tightened >= uni - 1e-8
        if rank == 0:
            rho = row_energy
        else:
            rho = row_energy - np.sum((W @ vh[:rank].T) ** 2, axis=1)
        stored = np.argsort(-rho, kind="mergesort")[:k]
        fit = fit_shape(W, rank, stored)
        assert tightened <= fit.error + 1e-5


def test_heavy_rows_beat_truncated_svd():
    rng = np.random.default_rng(0)
    n, m = 40, 16
    background = rng.normal(size=(n, 2)) @ rng.normal(size=(2, m))
    W = background.copy()
    W[[5, 17, 28]] = rng.normal(size=(3, m)) * 8.0
    W = W + rng.normal(size=(n, m)) * 0.05
    budget = cost(n, m, 2, 0)
    s2 = np.linalg.svd(W, compute_uv=False) ** 2
    svd_error = float(np.sum(s2[2:]))
    res = dslr.decompose(W, mode="budget", budget=budget)
    assert res.cost <= budget
    assert res.error < svd_error


def test_grid_never_worse_than_affordable_svd():
    rng = np.random.default_rng(4)
    W = rng.normal(size=(14, 9))
    n, m = W.shape
    T = _tails(W)
    rmax = min(n, m)
    for rank in range(0, rmax + 1):
        budget = cost(n, m, rank, 0)
        best_svd = min(T[r] for r in range(rmax + 1) if cost(n, m, r, 0) <= budget)
        res = dslr.decompose(W, mode="budget", budget=budget)
        assert res.cost <= budget
        assert res.error <= best_svd + 1e-6
    for rank in (0, 1, rmax):
        res = dslr.decompose(W, shape=(rank, 0))
        assert abs(res.error - T[rank]) <= 1e-6


def test_tau2_on_pure_noise():
    rng = np.random.default_rng(5)
    sigma = 0.7
    W = rng.normal(scale=sigma, size=(80, 40))
    res = dslr.decompose(W, mode="ebic")
    assert math.isfinite(res.tau2)
    assert res.tau2 > 0.0
    ratio = res.tau2 / sigma ** 2
    assert 1.0 / 3.0 <= ratio <= 3.0


def test_ebic_no_nans():
    rng = np.random.default_rng(6)
    W = rng.normal(size=(13, 6))
    res = dslr.decompose(W, mode="ebic")
    assert np.isfinite(res.reconstruction).all()
    assert np.isfinite(res.S).all()
    assert np.isfinite(res.L).all()
    for value in (res.error, res.floor, res.qmax, res.lam, res.tau2):
        assert math.isfinite(value)
    n, m = W.shape
    assert 0 <= res.rank <= min(n, m)
    assert 0 <= res.k <= n
    assert res.stored_rows.shape == (res.k,)
    assert res.cost == cost(n, m, res.rank, res.k)
    N = (min(n, m) + 1) * (n + 1)
    expected_lam = res.tau2 * (math.log(n * m) + 2.0 * math.log(N))
    assert abs(res.lam - expected_lam) <= 1e-8

    priced = dslr.decompose(W, lam=1.5)
    assert priced.lam == 1.5
    assert np.isfinite(priced.reconstruction).all()

    import jax.numpy as jnp

    from_jax = dslr.decompose(jnp.asarray(W), mode="ebic")
    assert np.isfinite(from_jax.reconstruction).all()
    assert math.isfinite(from_jax.qmax)
    assert from_jax.tau2 > 0.0
