import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time

# %matplotlib inline
# %matplotlib auto

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.95, 's' : 80, 'linewidths':0}

import numpy as np
cmat = np.load('tests/hero.npy')

cmat = cmat.T

# from cmat2aset import cmat2aset
# aset = cmat2aset(cmat)

plt.ion()  # plt.show()

n_row, n_col = cmat.shape
tset = [*zip(range(n_col), cmat.argmax(axis=0), cmat.max(axis=0))]
tset = np.array(tset)

# tset1 = [*zip(range(n_col), cmat.argmax(axis=0), cmat.max(axis=0))]
# tset1 = np.array(tset1)

# https://stackoverflow.com/questions/38246559/how-to-create-a-heat-map-in-python-that-ranges-from-green-to-red
from  matplotlib.colors import LinearSegmentedColormap
# cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
cmap=LinearSegmentedColormap.from_list('rg',["r", "g", "b"], N=20)
# cmap=LinearSegmentedColormap.from_list('rg',["darkred","red","lightcoral","white", "palegreen","green","darkgreen"], N=20)
cmap=LinearSegmentedColormap.from_list('rg',["darkred","red","lightcoral", "palegreen","green","darkgreen"], N=10)
cmap=LinearSegmentedColormap.from_list('rg',["darkred","red", "green","darkgreen"], N=10)
cmap=LinearSegmentedColormap.from_list('rg',["red", "green"], N=10)
cmap=LinearSegmentedColormap.from_list('rg',["red", "green","darkgreen"], N=10)
cmap=LinearSegmentedColormap.from_list('rg',["red","darkgreen","green"], N=8)

cmap = LinearSegmentedColormap.from_list('rg',["red","darkgreen","green"], N=6)

def plot_pset(pset, scale=1, cmap=cmap, alpha=1):
    # fig, ax = plt.subplots()
    # ax.scatter...
    pset = np.array(pset)
    plt.figure()
    plt.scatter(
        pset.T[0],
        pset.T[1],
        s=scale * pset.T[2],
        vmin=0,
        # cmap='viridis',
        cmap=cmap,
        # cmap='gray',
        # c=df.z,
        c=pset.T[2],
        alpha=alpha,
    )
    plt.grid()
    plt.colorbar()

plot_pset(tset, cmap='Greens')
plot_pset(tset)
plot_pset(tset, cmap='gist_earth_r')
plot_pset(tset, cmap='PuBuGn',)
plot_pset(tset, cmap='cool_r',)
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

plt.figure()

plt.scatter(tset.T[0], tset.T[1], c=tset.T[2], s=100*tset.T[2], cmap='gist_earth_r', alpha=0.6)
plt.colorbar()
plt.grid()

# from cmat2aset.gen_pset import gen_pset, _gen_pset, c_euclidean
from cmat2aset.gen_pset import gen_pset, _gen_pset
from cmat2aset.gen_pset import c_euclidean as c_euclidean0

# C:\syncthing\mat-dir\playground\clustering-dbscan-hdbscan\hdbscan-pad.py

eps = 10
min_samples = 2

labels = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=c_euclidean).fit(tset).labels_
# labels = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=lambda x, y: c_euclidean(x, y, delta=10)).fit(tset).labels_
palette = sns.color_palette('deep', np.unique(labels).max() + 1)
# colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
colors = [palette[x] if x >= 0 else (0.95, 0.95, 0.95) for x in labels]
plt.figure()
plt.scatter(tset.T[0], tset.T[1], c=colors, **plot_kwds)
plt.colorbar()

# eps=12, min_samples=6, metric='euclidean'
# eps=12, min_samples=3, metric=None  # c_euclidean

plot_kwds = {'alpha': 0.5,  'linewidths': 0}

from about_time import about_time
from icecream import ic

def plot_labels(tset, eps=12, min_samples=3, metric=None, cmap='PuBuGn', scale=1, alpha=1):
    if metric is None:
        metric = c_euclidean
    tset = np.array(tset)
    with about_time() as d:
        labels = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(tset).labels_
    ic(d.duration)
    # labels = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=lambda x, y: c_euclidean(x, y, delta=10)).fit(tset).labels_
    # colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    # palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    # colors = [palette[x] if x >= 0 else (0.95, 0.95, 0.95) for x in labels]

    colors = [elm if labels[idx] >= 0 else 1 for idx, elm in enumerate(tset.T[2])]

    tset0 = tset[labels >= 0]

    plt.figure()
    plt.scatter(
        tset0.T[0],
        tset0.T[1],
        s=scale * tset0.T[2],
        vmin=0,
        # cmap='viridis',
        cmap=cmap,
        # cmap='gray',
        # c=df.z,
        c=tset0.T[2],
        alpha=alpha,
    )

    # plt.scatter(tset.T[0], tset.T[1], c=colors, s=30*tset.T[2], **plot_kwds)

    r = 1 - labels.tolist().count(-1)/len(labels)
    r = math.trunc(r * 100) / 100
    plt.title(f'eps={eps}, min_s={min_samples}\nr={r}, thr={thr}')
    plt.colorbar()
    ic(r)
    plt.grid()
    return labels

thr = 9
labels = plot_labels(tset, eps=1000, min_samples=2, metric=lambda x, y: c_euclidean(x, y, thr=thr))  #  r: 0.18
# min_samples=2 thr = 10: r = 0.21
# min_samples=3 thr = 12: r = 0.27  some outliers
# min_samples=3 thr = 10: r = 0.14  no outliers

thr = 12; labels = plot_labels(tset, eps=1000, min_samples=3, metric=lambda x, y: c_euclidean(x, y, thr=thr))  # r .18, some outliers
thr = 11; labels = plot_labels(tset, eps=1000, min_samples=3, metric=lambda x, y: c_euclidean(x, y, thr=thr))  # r .17, few outliers
thr = 11; labels = plot_labels(tset, eps=1000, min_samples=4, metric=lambda x, y: c_euclidean(x, y, thr=thr))  # r .13, no outliers
thr = 13; labels = plot_labels(tset, eps=1000, min_samples=4, metric=lambda x, y: c_euclidean(x, y, thr=thr))  # r: 0.16, no outliers
thr = 15; labels = plot_labels(tset, eps=1000, min_samples=4, metric=lambda x, y: c_euclidean(x, y, thr=thr))  # r: 0.2, few outliers

labels0 = plot_labels(tset, metric=c_euclidean0)
labels = plot_labels(tset, metric=c_euclidean)

tset_split = np.split(tset, 3, axis=0)


# C:\syncthing\mat-dir\pypi-projects\tinybee-aligner\tinybee\gen_iset.py
# interpolate_pset.py

from cmat2aset.interpolate_pset import interpolate_pset

_ = '''
df_tset = pd.DataFrame(tset, columns=["x", "y", "cos"])
cset = df_tset[labels > -1].to_numpy()

# sort cset
_ = sorted(cset.tolist(), key=lambda x: x[0])
# '''

labels = plot_labels(tset, eps=12, min_samples=6)
cset = [el.tolist() for i, el in enumerate(tset) if labels[i] > -1]
# cset == sorted(cset, key=lambda x: x[0])

iset = interpolate_pset(cset, cmat.shape[1])
iset0 = np.array(iset)

plt.scatter(iset0.T[0], iset0.T[1], alpha=0.005)

# see also C:\mat-dir\snippets\note-to-self-cmat2aset.txt

---

plot_pset(tset)

def plot_labels(tset, eps=20, min_samples=3, metric=None, cmap='PuBuGn', scale=1, alpha=1):
    if metric is None:
        metric = c_euclidean
    tset = np.array(tset)
    with about_time() as d:
        labels = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(tset).labels_
        # labels = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=lambda x, y: metric(x,y, thr=thr)).fit(tset).labels_
    ic(d.duration)

    colors = [elm if labels[idx] >= 0 else 1 for idx, elm in enumerate(tset.T[2])]

    tset0 = tset[labels >= 0]

    plt.figure()
    plt.scatter(
        tset0.T[0],
        tset0.T[1],
        s=scale * tset0.T[2],
        vmin=0,
        # cmap='viridis',
        cmap=cmap,
        # cmap='gray',
        # c=df.z,
        c=tset0.T[2],
        alpha=alpha,
    )

    r = 1 - labels.tolist().count(-1)/len(labels)
    r = math.trunc(r * 100) / 100
    plt.title(f'eps={eps}, min_s={min_samples}\nr={r}, thr={thr}')
    plt.colorbar()
    ic(r)
    plt.grid()
    return labels

thr = 3; labels1 = plot_labels(tset, metric=c_euclidean)
thr = 19; labels1 = plot_labels(tset, metric=c_euclidean)  # eps=12, min_samples=3: r .12
thr = 19; eps=12; min_samples=2; labels1 = plot_labels(tset, min_samples=min_samples, metric=c_euclidean)  # r .19
thr = 19; eps=24; min_samples=3; labels1 = plot_labels(tset, eps=eps, min_samples=min_samples, metric=c_euclidean)  # r .26

tset1 = tset[labels1 > -1]
tset1_top10 = np.array(tset1[np.argpartition(-tset1.T[2], 10)[:10]])
tset1_top10a = np.array(sorted(tset1_top10, key=lambda x: x[0]))

plot_pset(tset1)

plot_pset(tset1_top10)
plt.plot(tset1_top10a.T[0], tset1_top10a.T[1], c='red', alpha=0.7)

thr = 19; eps=24; min_samples=2; labels1 = plot_labels(tset, eps=eps, min_samples=min_samples, metric=c_euclidean)  # r 0.41
thr = 19; eps=maxsize; min_samples=3; labels1 = plot_labels(tset, eps=eps, min_samples=min_samples, metric=c_euclidean)  # r 1.

thr = 19; eps=20; min_samples=3; labels1 = plot_labels(tset, eps=eps, min_samples=min_samples, metric=c_euclidean)  # r .26
thr = 19; eps=20; min_samples=4; labels1 = plot_labels(tset, eps=eps, min_samples=min_samples, metric=c_euclidean)  # r .21
thr = 19; eps=20; min_samples=5; labels1 = plot_labels(tset, eps=eps, min_samples=min_samples, metric=c_euclidean)  # r .16
thr = 19; eps=20; min_samples=6; labels1 = plot_labels(tset, eps=eps, min_samples=min_samples, metric=c_euclidean)  # r .1

# split to about 600

tset_s = np.array_split(tset, math.trunc(1883/600 + 0.5))
labels_s0 = plot_labels(tset_s[0])  # 0.22
labels_s1 = plot_labels(tset_s[1])  # 0.31
labels_s2 = plot_labels(tset_s[2])  # .28

labels_s0 = plot_labels(tset_s[0], min_samples=5)  # .14
labels_s1 = plot_labels(tset_s[1], min_samples=5)  # .21
labels_s2 = plot_labels(tset_s[2], min_samples=5)  # .13

cset0 = tset_s[0][labels_s0 > -1]
cset1 = tset_s[1][labels_s1 > -1]
cset2 = tset_s[2][labels_s2 > -1]

cset0_2 = np.concatenate([cset0, cset1, cset2])
assert cset0_2.shape[1] == 3

cset0_2_top10 = cset0_2[np.argpartition(-cset0_2.T[2], 10)[:10]]
cset0_2_top10 = np.array(sorted(cset0_2_top10, key=lambda x: x[0]))

plt.figure(); plt.scatter(cset0_2_top10.T[0], cset0_2_top10.T[1], s=50*cset0_2_top10.T[2], c='red', alpha=.6)

'''
https://numpy.org/doc/stable/reference/generated/numpy.interp.html
    A simple check for xp being strictly increasing is:

       np.all(np.diff(xp) > 0)


'''
