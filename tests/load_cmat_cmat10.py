"""load cmat cmat10.

import matplotlib.pyplit as plt
import seaborn as sns

sns.set()
plt.ion()  # interactive plot

plt.clf(); sns.heatmap(cmat, cmap="gist_earth_r").invert_yaxis()

plt.clf(); sns.heatmap(cmat, cmap="viridis_r").invert_yaxis()

"""
import pickle
from pathlib import Path

cdir = Path(__file__).parent.resolve()

cmat = pickle.load(open(cdir / "cos_matrix.pkl", "rb"))
cmat10 = pickle.load(open(cdir / "cos_matrix10.pkl", "rb"))
