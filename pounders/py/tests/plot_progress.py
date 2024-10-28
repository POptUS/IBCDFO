import numpy as np
import matplotlib.pyplot as plt
import glob

# Find all files starting with "first_example" and ending with ".npy"
file_pattern = "./first_example*.npy"
file_list = glob.glob(file_pattern)

# Check if any files were found
if not file_list:
    print("No files found with the specified pattern.")
else:
    min_value = float('inf')

    # First pass: determine the minimum "H" value across all files
    for file in file_list:
        data = np.load(file, allow_pickle=True).item()
        if "H" in data:
            min_value = min(min_value, np.min(data["H"]))
    
    # Shift to make all values positive
    shift_value = abs(min_value) + 1e-16

    # Second pass: load and plot the shifted "H" values
    for file in file_list:
        data = np.load(file, allow_pickle=True).item()
        if "H" in data:
            H_values = data["H"] + shift_value  # Ensure all values are positive
            plt.semilogy(H_values, label=file)
        else:
            print(f"'H' not found in {file}")

    # Add plot labels and title
    plt.xlabel("Iteration or Data Point Index")
    plt.ylabel("Shifted H Values (Log Scale)")
    plt.title("Progress of Shifted 'H' Values from first_example Files")
    plt.legend()
    plt.savefig("first_plot.png", dpi=300)
