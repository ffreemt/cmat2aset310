"""Gen pset from cmat. Find pairs for a given cmat.

tinybee.find_pairs.py with fixed estimator='dbscan' eps=eps, min_samples=min_samples
"""
# pylint: disable=too-many-locals, unused-import, invalid-name, too-many-arguments, broad-except, too-many-branches, too-many-statements, too-many-statements

from typing import Callable, List, Optional, Tuple, Union

from sys import maxsize
from math import atan2, degrees, dist  # type: ignore
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import logzero
from logzero import logger
from set_loglevel import set_loglevel

from cmat2aset310.cmat2tset import cmat2tset
from cmat2aset310.interpolate_pset import interpolate_pset
from cmat2aset310.get_angle import get_angle

logzero.loglevel(set_loglevel())


# def c_euclidean(x, y, scale=10.):
def c_euclidean(x, y, delta=3.):
    """Calculate customized Euclidean distance.

    Args:
        x, y: coordinates
        delta: 0..45

    Returns:
        customized Euclidean distance:
        points in second quadrant/fourth quadrant set to sys.maxsize

    >>> 1.7 > c_euclidean([0, 0], [1, 1]) > 1.4
    True
    >>> 1.7 > c_euclidean([0, 0], [-1, -1]) > 1.4
    True
    >>> c_euclidean([0, 0], [-1, 1]) == maxsize
    True
    >>> c_euclidean([0, 0], [1, -1]) == maxsize
    True
    """
    _ = """
    if x[0] == y[0] or x[1] == y[1]:
        return maxsize

    if abs(x[0] - y[0]) >= scale * abs(x[1] - y[1]) or abs(x[1] - y[1]) > scale * abs(
        x[0] - y[0]
    ):
        return maxsize
    """
    if delta < 0:
        delta = 3
    if delta > 45:
        delta = 3
    angle = get_angle(y, x)
    # if angle > 90 - delta and angle < 180 + delta:  # second quadrant
    if 90 < angle < 180 + delta:  # second quadrant
        return maxsize
    if angle < delta or angle > 270 - delta:  # fourth quadrant
        return maxsize

    return dist(x, y)


def _gen_pset(
    cmat1: Union[List[List[float]], np.ndarray, pd.DataFrame],
    eps: Optional[float] = None,  # 10/30,
    min_samples: Optional[int] = None,  # 6/3,
    # metric: Union[str, Callable]="euclidean",
    metric: Optional[Union[str, Callable]] = None,  # c_euclidean,
    delta: float = 7,
    # verbose: Union[bool, int] = False,
    # ) -> List[Tuple[int, int, Union[float, str]]]:
) -> List[Tuple[Union[float, str], Union[float, str], Union[float, str]]]:
    """Gen pset from cmat.

    Find pairs for a given cmat.

    Args:
        cmat: correlation/similarity matrix
        eps: min epsilon for DBSCAN (10)
        min_samples: minimum # of samples for DBSCAN (6)
        delta: tolerance (7)

    Returns:
        pairs + "" or metric (float)

    dbscan_pairs' setup
        if eps is None:
            eps = src_len * .01
            if eps < 3:
                eps = 3
        if min_samples is None:
            min_samples = tgt_len / 100 * 0.5
            if min_samples < 3:
                min_samples = 3

    def gen_eps_minsamples(src_len, tgt_len):
        eps = src_len * .01
        if eps < 3:
            eps = 3

        min_samples = tgt_len / 100 * 0.5
        if min_samples < 3:
            min_samples = 3
        return {"eps": eps, "min_samples": min_samples}

    """
    if metric is None:
        metric = c_euclidean
    elif metric not in [
        "l2",
        "nan_euclidean",
        "manhattan",
        "l1",
        "sokalsneath",
        "hamming",
        "sqeuclidean",
        "jaccard",
        "seuclidean",
        "yule",
        "dice",
        "euclidean",
        "cosine",
        "rogerstanimoto",
        "cityblock",
        "wminkowski",
        "russellrao",
        "chebyshev",
        "precomputed",
        "braycurtis",
        "correlation",
        "minkowski",
        "matching",
        "canberra",
        "sokalmichener",
        "mahalanobis",
        "kulsinski",
        "haversine",
        c_euclidean,
    ]:
        raise Exception("Invalid metric param")

    # adjuset default min_samples -- euclidean: 6, c_euclidean: 3
    if min_samples is None:
        if callable(metric):
            min_samples = 3
        # if metric in ["euclidean"]:
        else:
            min_samples = 6

    if eps is None:
        if callable(metric):
            eps = 20
        # if metric in ["euclidean"]:
        else:
            eps = 10

    # if isinstance(cmat, list):
    cmat = np.array(cmat1)

    src_len, tgt_len = cmat.shape

    # iset = gen_iset(cmat, verbose=verbose, estimator=estimator)
    if callable(metric):
        tset = [
            *zip(
                range(cmat.shape[1]),
                cmat.argmax(axis=0).tolist(),
                cmat.max(axis=0).tolist(),
            )
        ]
        # shape = cmat.shape
        # scale = max(3, src_len / tgt_len, tgt_len / src_len)

        # if delta is None:
        delta = degrees(atan2(tgt_len, src_len)) / 10
        logger.debug(" delta: %s", delta)
        logger.info(" delta: %s", delta)
        delta = 3
        logger.info("fixed delta: %s", delta)

        labels = (
            DBSCAN(
                eps=eps,
                min_samples=min_samples,
                # metric=lambda x, y: c_euclidean(x, y, scale=scale)
                # metric=lambda x, y: metric(x, y),  # fixed delta=3
                metric=lambda x, y: metric(x, y, delta=delta),
            )
            .fit(tset)
            .labels_
        )
    # if metric in ["euclidean"]:
    else:
        # tset = cmat2tset(cmat)
        tset = cmat2tset(cmat).tolist()

        logger.debug("tset: %s", tset)
        labels = (
            DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(tset).labels_
        )

    df_tset = pd.DataFrame(tset, columns=["x", "y", "cos"])
    cset = df_tset[labels > -1].to_numpy()

    # sort cset
    _ = sorted(cset.tolist(), key=lambda x: x[0])

    if not _:
        raise Exception(
            "No cset collected, meaning cmat is invalid or eps/min_samples "
            f"{eps}/{min_samples} not properly set"
        )

    iset = interpolate_pset(_, tgt_len)

    # *_, ymax = zip(*tset)
    # ymax = list(ymax)
    # low_ = np.min(ymax) - 1  # reset to minimum_value - 1

    buff = [(-1, -1, ""), (tgt_len, src_len, "")]

    # for idx, tset_elm in enumerate(tset):
    for tset_elm in tset:
        # logger.debug("buff: %s", buff)
        # postion max in ymax and insert in buff
        # if with range given by iset+-delta and
        # it's valid (do not exceed constraint
        # by neighboring points

        # argmax = int(np.argmax(ymax))

        # logger.debug("=== %s,%s === %s", _, argmax, tset[_])
        logger.debug("=== %s === %s", _, tset_elm)

        # ymax[_] = low_
        # elm = tset[argmax]
        # elm0, *_ = elm

        elm0, *_ = tset_elm

        # position elm in buff
        idx = -1  # for making pyright happy
        for idx, loc in enumerate(buff):
            if loc[0] > elm0:
                break
        else:
            idx += 1  # last

        # insert elm in for valid elm
        # (within range inside two neighboring points)

        # pos = int(tset[argmax][0])
        pos = int(tset_elm[0])
        logger.debug(" %s <=> %s ", tset_elm, iset[pos])

        # if abs(tset[argmax][1] - iset[pos][1]) <= delta:
        if abs(tset_elm[1] - iset[pos][1]) <= delta:
            if tset_elm[1] > buff[idx - 1][1] and tset_elm[1] < buff[idx][1]:
                buff.insert(idx, tset_elm)
                logger.debug("idx: %s, tset_elm: %s", idx, tset_elm)
            else:
                logger.debug("\t***\t idx: %s, tset_elm: %s", idx, tset_elm)
        _ = """
        if abs(tset[loc][1] - iset[loc][1]) <= delta:
            if tset[loc][1] > buff[idx][1] and tset[loc][1] < buff[idx + 1][1]:
                buff.insert(idx + 1, tset[loc])
        # """

    # remove first and last entry in buff
    buff.pop(0)
    buff.pop()

    # return [(1, 1, "")]
    return [(int(elm0), int(elm1), elm2) for elm0, elm1, elm2 in buff]


def gen_pset(
    cmat1: Union[List[List[float]], np.ndarray, pd.DataFrame],
    eps: Optional[float] = None,  # 10/12
    min_samples: Optional[int] = None,  # 6/3
    # metric="euclidean",
    metric: Optional[Union[str, Callable]] = None,  # c_euclidean,
    delta: float = 7,
    verbose: Union[bool, int] = False,
) -> List[Tuple[Union[float, str], Union[float, str], Union[float, str]]]:
    """Gen pset.

    Refer to _gen_pset.
    """
    del verbose

    if metric is None:
        metric = c_euclidean

    # adjuset default min_samples -- euclidean: 6, c_euclidean: 3
    if min_samples is None:
        if callable(metric):
            logger.info("Using %s", metric.__name__)
            min_samples = 3
        # if metric in ["euclidean"]:
        else:
            min_samples = 6

    if eps is None:
        if callable(metric):
            logger.info("Using %s", metric.__name__)
            eps = 12
        # if metric in ["euclidean"]:
        else:
            eps = 10

    gen_pset.min_samples = min_samples
    for min_s in range(min_samples):
        logger.debug(" min_samples, try %s", min_samples - min_s)
        try:
            pset = _gen_pset(
                cmat1,
                eps=eps,
                min_samples=min_samples - min_s,
                metric=metric,
                delta=delta,
            )
            logger.info(" min_samples: %s", min_samples - min_s)
            break
        except ValueError:
            continue
        except Exception as e:
            logger.error(e)
            continue
    else:
        # break should happen above when min_samples = 2
        raise Exception("bummer, this shouldn't happen, probably another bug")

    # store new min_samples
    gen_pset.min_samples = min_samples - min_s

    return pset
