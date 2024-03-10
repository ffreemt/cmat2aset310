"""Test cmat2aset.

sns.set_style("darkgrid")
pd.options.display.float_format = "{:,.2f}".format

res = cmat2aset(cmat)
res10 = cmat2aset(cmat10)

df = pd.DataFrame(res, columns=['x', 'y', 'z']).replace("", np.nan)
sns.heatmap(cmat, cmap="viridis_r").invert_yaxis()

plt.scatter(df.x, df.y, marker="v", cmap="virids_r")
# or
df.plot.scatter("x", "y", c="z")

sns.heatmap(cmat10, cmap="viridis_r").invert_yaxis()
plt.scatter(df10.x, df10.y, marker="v", cmap="virids_r")  # alpha = 0.1 .5 1

"""
# pylint: disable=broad-except
from cmat2aset310 import __version__, cmat2aset
from tests.load_cmat_cmat10 import cmat  # , cmat10


def test_version():
    """Test version."""
    assert __version__[:3] == "0.1"


def test_sanity():
    """Sanity check."""
    try:
        assert not cmat2aset()  # type: ignore
    except Exception:
        assert True


def test_cmat2aset_zero():
    """Test '0' (str) converted to 0 (in)."""
    # res = cmat2aset(cmat)
    res = cmat2aset(cmat, metric="euclidean")
    assert isinstance(res[0][0], int)
    assert isinstance(res[0][1], int)
    assert isinstance(res[0][2], float)
    assert sum(1 for el in res if el[2]) >= 45

def test_cmat2aset_minus_two():
    """Test '0' (str) converted to 0 (in)."""
    res = cmat2aset(cmat)
    assert isinstance(res[-2][0], str)
    assert isinstance(res[-2][1], int)
    assert isinstance(res[-2][2], str)
