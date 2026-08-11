# Demonstrates, with a SINGLE standalone call (no run_MSP, no optimization loop), that
# jaxnp_hash's tracing of h_one_norm_jax blows up combinatorially in the number of
# components of z that are tied at a kink (here, exactly 0) -- NOT in the dimension m
# of z itself. Each abs(z_i) call traces both sign branches when z_i is exactly 0, so
# the branch count in the resulting PathSet is ~2**(number of tied components).
#
# Contrast: the hand-coded h_one_norm deliberately collapses a near-zero residual to a
# single flat-gradient "0" branch specifically to avoid this 2**dim(z) blowup (see its
# own comment). jaxnp_hash's automatic tracing has no such collapse.
#
# Usage:
#   python demo_tie_break_blowup.py            # 10 tied components (~tens of seconds)
#   python demo_tie_break_blowup.py 14         # much slower -- doubles per +1
import sys
import time

sys.path.append("./jaxnp_hash/")

import numpy as np

import jan_example as je

NUM_TIED = int(sys.argv[1]) if len(sys.argv) > 1 else 10

z = np.zeros(NUM_TIED)  # worst case: every component is exactly at the abs() kink

print(f"Calling h_one_norm_jax once on an all-zero vector of length {NUM_TIED} " f"(every component tied at the kink)...")
print("Cost roughly doubles for each additional tied component, so this may take a while.")
t0 = time.time()
je.h_one_norm_jax(z)
print(f"Done in {time.time() - t0:.3f}s")
