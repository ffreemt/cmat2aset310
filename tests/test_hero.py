"""Test pset with hero.npy."""
import numpy as np
from cmat2aset310.gen_pset import gen_pset, _gen_pset

from pathlib import Path

cdir = Path(__file__).parent.resolve()
cmat_hero = np.load(Path(cdir, "hero.npy"))


def test_pset_c_euclidean():
    """Test c_euclidean."""
    pset = gen_pset(cmat_hero)  # 3.82 s  metric=c_euclidean
    assert len(pset) >= 129

    pset1 = gen_pset(cmat_hero, metric="euclidean")  # 20s
    assert len(pset1) >= 107
