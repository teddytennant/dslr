D-SLR (Disjoint Row-Sparse plus Low-Rank) is a closed-form drop-in for the truncated SVD: each row is stored verbatim or fit by a low-rank factor, never both. Paper: https://arxiv.org/abs/2610.10636.

Run the tests with `python -m pytest -q` from this directory. The tests use synthetic matrices only. This package does not reproduce the paper's LLM-embedding or hyperspectral experiments. The Marchenko-Pastur factor in the noise estimate is a numerical integral. The optional shortlist verification is not run, and the split floor uses the round-robin partitions, not the hybrid strata variant.
