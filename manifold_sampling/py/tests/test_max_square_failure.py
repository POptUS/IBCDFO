import numpy as np
from ibcdfo.manifold_sampling.h_examples import pw_maximum_squared

F = np.array(
    [
        -40708.6555510835,
        -33467.081758611654,
        -27653.379955714125,
        -22948.884742964157,
        -19136.688304013853,
        -16038.602022131898,
        -13492.950856289473,
        -11396.96000675333,
        -9666.566166808527,
        -8230.53545148878,
        -7034.804454170586,
        -6033.984782865407,
        -5193.98818276539,
        -4485.741184570553,
        -3886.9666353384414,
        -3379.015497652459,
    ]
)

hfun = pw_maximum_squared

h_dummy1, grad_dummy1, hash_dummy = hfun(F)
h_dummy2, grad_dummy2 = hfun(F, hash_dummy)

assert h_dummy1 == h_dummy2, "hfun values don't agree when " + hfun.__name__ + " is re-called with the same inputs"
assert np.all(grad_dummy1 == grad_dummy2), "hfun gradients don't agree when " + hfun.__name__ + " is being re-called with the same inputs"
