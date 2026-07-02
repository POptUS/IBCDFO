import numpy as np
import sys

sys.path.append('./jaxnp_hash/')

from jan_example import h_max_gamma_over_KY_jax as jan_hfun
from ibcdfo.manifold_sampling import h_max_gamma_over_KY as old_hfun


a = np.load("jans_msp_output_1.npz", allow_pickle=True)
b = np.load("old_msp_output_1.npz", allow_pickle=True)

for key in ["X", "F", "h_msp"]:
    print(f"\n{key}:")
    print("  same shape:", a[key].shape == b[key].shape)
    print("  exactly equal:", np.array_equal(a[key], b[key]))
    print("  allclose:", np.allclose(a[key], b[key], rtol=1e-12, atol=1e-12, equal_nan=True))

    if a[key].shape == b[key].shape:
        diff = a[key] - b[key]
        print("  max abs diff:", np.nanmax(np.abs(diff)))


diff_h = a["h_msp"] - b["h_msp"]
rows = np.unique(np.where(diff_h)[0])

print("\nRows where h_msp differs:", rows)

for row in rows:
    Frow = a["F"][row]

    print(f"\n=== row {row} ===")
    print("Frow:", Frow)
    print("saved jan h_msp:", a["h_msp"][row])
    print("saved old h_msp:", b["h_msp"][row])
    print("saved diff:", a["h_msp"][row] - b["h_msp"][row])

    jan_val = jan_hfun(Frow)
    old_val = old_hfun(Frow)

    print("jan_hfun(Frow):", jan_val)
    print("old_hfun(Frow):", old_val)
