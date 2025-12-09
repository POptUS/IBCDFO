import os
import glob
import numpy as np

# Adjust this import to match where formquad actually lives in your codebase
from ibcdfo.pounders.formquad import formquad

def compare_formquad_results(results_dir="formquad_results"):
    pattern = os.path.join(results_dir, "formquad_out_row=*call=*.npy")

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No .npy files found in {results_dir}")
        return

    all_ok = True

    for fname in files:
        print(f"\nChecking {fname}")
        data = np.load(fname, allow_pickle=True).item()  # dict

        # Inputs
        X      = data["X"]
        Res    = data["Res"]
        delta  = data["delta"]
        xk_in  = data["xk_in"]
        np_max = data["np_max"]
        Par    = data["Par"]
        last_arg = data["last_arg"]

        # Re-run formquad with saved inputs
        Mdir_new, mp_new, valid_new, Gres_new, H_new, Mind_new = formquad(X, Res, delta, xk_in, np_max, Par, last_arg)

        new_outputs = {
            "Mdir": Mdir_new,
            "mp": mp_new,
            "valid": valid_new,
            "Gres": Gres_new,
            "H": H_new,
            "Mind": Mind_new,
        }

        # Compare only outputs that exist in the saved dictionary
        keys_to_check = ["Mdir", "mp", "valid", "Gres", "H", "Mind"]
        file_ok = True

        for key in keys_to_check:
            if key not in data:
                continue  # this output not saved for this call

            saved = data[key]
            new   = new_outputs[key]

            # Handle numpy arrays vs scalars/other types
            if isinstance(saved, np.ndarray) or isinstance(new, np.ndarray):
                # Use allclose for floats; handles int arrays as well
                equal = np.allclose(saved, new, rtol=1e-12, atol=1e-14)
            else:
                equal = (saved == new)

            if not equal:
                print(f"  MISMATCH for {key}:")
                if isinstance(saved, np.ndarray) and isinstance(new, np.ndarray):
                    print(f"    saved.shape = {saved.shape}, new.shape = {new.shape}")
                    diff_norm = np.linalg.norm(saved.astype(float) - new.astype(float))
                    print(f"    ||saved - new||_2 = {diff_norm:.3e}")
                else:
                    print(f"    saved = {saved!r}")
                    print(f"    new   = {new!r}")
                file_ok = False
                all_ok = False

        if file_ok:
            print("  All checked outputs match for this file.")

    if all_ok:
        print("\nAll formquad results matched for every file.")
    else:
        print("\nSome mismatches were found. See messages above for details.")


if __name__ == "__main__":
    compare_formquad_results()

