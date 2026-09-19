import numpy as np
from branch_extended_AD import path_key
from scipy.spatial.distance import cdist


def _as_list(x):
    """Return x as a flat Python list without requiring entries to be hashable."""
    if x is None:
        return []

    if isinstance(x, np.ndarray):
        return x.ravel().tolist()

    if isinstance(x, (list, tuple, set)):
        return list(x)

    try:
        return list(x)
    except TypeError:
        return [x]


def _extend_unique(existing, new_items):
    """Add entries from new_items to existing without requiring hashability.

    Hand-coded hfuns represent each Hash entry as a plain (hashable) string;
    jax hfuns' Hash entries are branch_extended_AD paths (lists of _TraceNode),
    which define __eq__ but not __hash__ and are therefore unhashable on their
    own. branch_extended_AD.path_key gives an O(1)-hashable proxy for both
    kinds of entry, so this dedup stays O(n) regardless of which kind of hfun
    produced Hash -- without it, a single evaluated point whose Hash
    combinatorially explodes (e.g. many near-simultaneous ties in a
    censored-L1-type hfun) would make an O(n^2) equality scan the dominant
    cost of the whole algorithm.
    """
    out = _as_list(existing)
    seen = {path_key(item) for item in out}

    for item in _as_list(new_items):
        key = path_key(item)
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


def _maybe_as_array(x):
    """
    Convert to np.asarray when possible, but keep object lists if NumPy-style
    conversion/indexing is not appropriate.
    """
    try:
        arr = np.asarray(x)
    except Exception:
        return x

    # If this became an object array of custom objects, a Python list is safer.
    if arr.dtype == object:
        return _as_list(x)

    return arr


def _take_by_indices(x, indices):
    """Index either a NumPy array or a Python list by a list/array of indices."""
    if isinstance(x, np.ndarray):
        return x[indices]

    x_list = _as_list(x)
    return [x_list[int(j)] for j in indices]


def choose_generator_set(X, Hash, xkin, nf, delta, F, hfun):
    Act_Z_k = Hash[xkin]

    hxkin, _ = hfun(F[xkin, :], Act_Z_k)
    XkDist = cdist(X[: nf + 1], X[xkin : xkin + 1], metric="chebyshev")
    delta1 = delta * (1 + 1e-8)
    delta2 = min(1, delta) ** 2 * (1 + 1e-8)

    for i, XkDi in enumerate(XkDist):
        if XkDi <= delta1:
            Act_tmp = Hash[i]
            h_i, _ = hfun(F[xkin], Act_tmp)
            if h_i[0] <= hxkin[0] or XkDi <= delta2:
                if i != xkin:
                    Act_Z_k = _extend_unique(Act_Z_k, Act_tmp)

    Act_Z_k = _maybe_as_array(Act_Z_k)

    f_k, D_k = hfun(F[xkin], Act_Z_k)

    try:
        unique_indices = np.unique(D_k, axis=1, return_index=True)[1]
        unique_indices = np.sort(unique_indices)

        D_k = D_k[:, unique_indices]
        Act_Z_k = _take_by_indices(Act_Z_k, unique_indices)
        f_k = _take_by_indices(f_k, unique_indices)
    except Exception:
        # Fallback: if D_k is not in a NumPy-compatible format, skip this
        # deduplication. The activity accumulation above has already removed
        # duplicates based on equality of Act_Z_k entries.
        pass

    return D_k, Act_Z_k, f_k
