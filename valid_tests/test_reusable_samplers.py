from src.small_visual_world.sampling import stratified_adaptation_indices
from src.medium_visual_world.sampling import balanced_indices


class AdaptRows:
    labels = [0] * 64 + [1] * 64


def test_stratified_adaptation_reproducible():
    train, dev = stratified_adaptation_indices(AdaptRows(), seed=1919, dev_per_class=16)
    assert len(train) == 96 and len(dev) == 32
    assert set(train).isdisjoint(dev)
    assert sum(AdaptRows.labels[i] for i in train) == 48
    assert sum(AdaptRows.labels[i] for i in dev) == 16
    assert (train, dev) == stratified_adaptation_indices(AdaptRows(), seed=1919)
