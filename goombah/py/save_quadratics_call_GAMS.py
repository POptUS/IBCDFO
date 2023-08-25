# A general Python-to-GAMS interface.
import numpy as np
import subprocess

def save_quadratics_call_GAMS(H, g, b, Low, Upp, x0, x1, val_at_x0, GAMS_options, hfun):
    n, P = g.shape

    # Loop over solvers
    for i in GAMS_options['solvers']:
        solver_val = i

        # Put problem data to a gdx file
        # wgdx('quad_model_data', Ns, Ps, Hs_mod, gs_mod, bs_mod, x0s, x1s, solver, Lows, Upps)
        # print('Data written to GDX file quads_model_data.gdx')

        # # Copy the template gams file
        # subprocess.run(['cp', GAMS_options['file'], f'./TRSP_{i}.gms'])

        # # Perform the gams run
        # flag = subprocess.call(['gams', f'TRSP_{i}.gms', 'lo=2'])

        # assert flag == 0, f'gams run failed: rc = {flag}'
        # assert os.path.exists(solGDX), f'Results file {solGDX} does not exist after gams run'

        # print(f'TRSP_{i}.gms finished')

        # # Now get the outputs from the GDX file produced by the GAMS run
        # rs = {'name': 'modelStat', 'form': 'full'}
        # r = rgdx(solGDX, rs)
        # modelStat[i] = r['val']

        # rs['name'] = 'solveStat'
        # r = rgdx(solGDX, rs)
        # solveStat[i] = r['val']

        # rs['name'] = 'tau'
        # rs['field'] = 'l'
        # r = rgdx(solGDX, rs)
        # obj_vals_GAMS[i] = r['val']

        # rs['name'] = 'x'
        # rs['uels'] = Ns['uels']
        # r = rgdx(solGDX, rs)
        # x = r['val']
        # allx[i, :] = x

    # Just randomly try values
    num_rand_samp = 1000
    obj_vals_PYTHON = np.zeros(num_rand_samp)
    allx = np.random.uniform(x0 + Low, x0 + Upp, (num_rand_samp, n))
    z = np.zeros(P)
    for j, x  in enumerate(allx):
        for i in range(P):
            z[i] = 0.5 * (x - x0) @ H[:, :, i] @ (x - x0) + (x - x0) @ g[:, i] + b[i]
        obj_vals_PYTHON[j] = hfun(z)[0]

    val_at_new = np.min(obj_vals_PYTHON)
    ind = np.argmin(obj_vals_PYTHON)
    s_k = allx[ind, :] - x0
    pred_dec = val_at_x0 - val_at_new

    if pred_dec < 0:
        pred_dec = 0

    return s_k, pred_dec
