#!/usr/bin/env python3

import dill,sys
import numpy as np

from ibcdfo.pounders import bmpts, formquad


# Pounders with and without Jax are about to diverge on a BenDFO problem.
# They do so at this formquad call. Perhaps this is not a big deal, but I
# wanted to note the following: 

print("\nGet these two data files before running: \n www.mcs.anl.gov/~jlarson/my_var_before_formquad_ibcdfo.pkl \n www.mcs.anl.gov/~jlarson/my_var_before_formquad_jax.pkl\n")

with open("my_var_before_formquad_ibcdfo.pkl", "rb") as f:
    data1 = dill.load(f)

X1 = data1["X"]
F1 = data1["F"]
nf1 = data1["nf"]
delta1 = data1["delta"]
xk_in1 = data1["xk_in"]
Model1 = data1["Model"]

# ---- Call formquad on the "IBCDFO only" dataset ----
[Mdir1, mp1, valid1, G1, H1, Mind1] = formquad(X1[: nf1 + 1, :], F1[: nf1 + 1, :], delta1, xk_in1, Model1["np_max"], Model1["Par"], 1)

# ---- Load the second pickle file ----
with open("my_var_before_formquad_jax.pkl", "rb") as f:
    data2 = dill.load(f)

X2 = data2["X"]
F2 = data2["F"]
nf2 = data2["nf"]
delta2 = data2["delta"]
xk_in2 = data2["xk_in"]
Model2 = data2["Model"]

# ---- Call formquad on the "Using Jax" dataset ----
[Mdir2, mp2, valid2, G2, H2, Mind2] = formquad(X2[: nf2 + 1, :], F2[: nf2 + 1, :], delta2, xk_in2, Model2["np_max"], Model2["Par"], 1)

print("X diff = ", np.linalg.norm(X1 - X2))
print("F diff = ", np.linalg.norm(F1 - F2))

print("Mind diff = ", np.array(Mind1) - np.array(Mind2))
print("Mdir diff = ", np.linalg.norm(Mdir1 - Mdir2))


