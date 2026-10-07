import random

from src.eval import corrupt, prf


def test_corrupt_swaps_to_a_different_chunk_and_records_truth():
    pairs = [(f"s{i}", 1 + i % 5) for i in range(200)]
    out = corrupt(pairs, n_chunks=5, rate=0.3, rng=random.Random(0))
    swapped = [c for c in out if c["corrupted"]]
    assert 40 <= len(swapped) <= 80
    assert all(c["cited"] != c["original"] for c in swapped)
    assert all(c["cited"] == c["original"] for c in out if not c["corrupted"])


def test_prf():
    assert prf([True, True, False, False], [True, False, True, False]) == (0.5, 0.5, 0.5)
    assert prf([False], [False]) == (0.0, 0.0, 0.0)


def test_stratified_folds_cover_everything_once_and_balance_positives():
    from src.eval import stratified_folds

    labels = [True] * 10 + [False] * 60
    folds = stratified_folds(labels, k=5, seed=0)
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(70))
    assert all(sum(labels[i] for i in f) == 2 for f in folds)
    assert folds == stratified_folds(labels, k=5, seed=0)   # deterministic


def test_fbeta_weights_precision():
    from src.eval import fbeta

    assert fbeta(1.0, 0.5, 0.5) > fbeta(0.5, 1.0, 0.5)
    assert fbeta(0.0, 0.0, 0.5) == 0.0
