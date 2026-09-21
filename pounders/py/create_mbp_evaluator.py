import numpy as np

from .constants import MBP_EVAL_SERIAL, MBP_EVAL_BATCH


def create_mbp_evaluator(mbp_eval):
    r"""
    Create a Python function that evaluates ``Ffun`` at the new model-building
    points (MBPs) needed to complete |pounders|' initial interpolation set.

    :param mbp_eval:
        * ``ibcdfo.pounders.MBP_EVAL_SERIAL`` - call ``Ffun`` once for each new
          model-building point
        * ``ibcdfo.pounders.MBP_EVAL_BATCH`` - call ``Ffun`` once with all new
          model-building points stacked as rows of a single NumPy array. 
          Users who want concurrent evaluation of model-building points should
          provide an ``Ffun`` that accepts a batch of points ``X_new`` with
          shape ``(batch_size, n)``, where each row is one point to evaluate,
          and returns an array with shape ``(batch_size, m)`` whose rows
          contain the corresponding ``Ffun`` outputs in the same order as the
          input points.

    :return: Python function with the interface

        .. code:: python

            F_new = evaluate_mbp(Ffun, X_new, m)

        where

        * **Ffun** is the user-provided blackbox function passed to |pounders|,
        * **X_new** is a ``batch_size``:math:`\times \np` NumPy array whose
          rows are the new model-building points to evaluate,
        * **m** is the dimension of the output of ``Ffun``, and
        * **F_new** is the ``batch_size``:math:`\times m` NumPy array of
          values of ``Ffun`` at the rows of **X_new**, in the same order.
    """
    if mbp_eval == MBP_EVAL_SERIAL:

        def __serial_evaluator(Ffun, X_new, m):
            F_new = np.zeros((X_new.shape[0], m))
            for i in range(X_new.shape[0]):
                F_new[i] = Ffun(X_new[i])
                if np.any(np.isnan(F_new[i])) or np.any(np.isinf(F_new[i])):
                    break
            return F_new

        return __serial_evaluator

    elif mbp_eval == MBP_EVAL_BATCH:

        def __batch_evaluator(Ffun, X_new, m):
            F_new = Ffun(X_new)
            assert F_new.shape == (X_new.shape[0], m)
            return F_new

        return __batch_evaluator

    raise ValueError(f"Unknown model-building point evaluator: {mbp_eval}")
