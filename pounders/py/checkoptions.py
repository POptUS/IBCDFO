from .._variable_checks import is_finite_real


def checkoptions(printf, spsolver, batched_Ffun, delta_max, delta_min, delta_inact, gamma_dec, gamma_inc, eta_1, hfun, combinemodels):
    """
    Raise an exception if POUNDERS option values do not meet the requirements.

    For information regarding the arguments, see :py:func:`ibcdfo.pounders.pounders`.
    """

    if not callable(spsolver):
        raise TypeError("Error: Options['spsolver'] is not a function")

    if not isinstance(batched_Ffun, bool):
        raise TypeError("Error: Options['batched_Ffun'] must be a boolean")

    if printf not in (0, 1, 2):
        raise ValueError(f"Error: Options['printf'] must be 0, 1, or 2 ({printf})")

    if not (is_finite_real(delta_max) and delta_max > 0.0):
        raise ValueError(f"Error: Options['delta_max'] must be a positive finite real ({delta_max})")
    if not (is_finite_real(delta_min) and delta_min >= 0.0):
        raise ValueError(f"Error: Options['delta_min'] must be a nonnegative finite real ({delta_min})")
    if not (is_finite_real(delta_inact) and 0.0 < delta_inact <= 1.0):
        raise ValueError(f"Error: Options['delta_inact'] must be a finite real in (0, 1] ({delta_inact})")
    if not (is_finite_real(gamma_dec) and 0.0 < gamma_dec < 1.0):
        raise ValueError(f"Error: Options['gamma_dec'] must be a finite real in (0, 1) ({gamma_dec})")
    if not (is_finite_real(gamma_inc) and gamma_inc >= 1.0):
        raise ValueError(f"Error: Options['gamma_inc'] must be a finite real >= 1 ({gamma_inc})")
    if not (is_finite_real(eta_1) and 0.0 < eta_1 < 1.0):
        raise ValueError(f"Error: Options['eta1'] must be a finite real in (0, 1) ({eta_1})")

    if not callable(hfun):
        raise TypeError("Error: Options['hfun'] must be a function")
    if not callable(combinemodels):
        raise TypeError("Error: Options['combinemodels'] must be a function")
