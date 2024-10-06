import A2_A0273761Y as mine
import A2_A0270412U as friend
import numpy as np

# Run your code
N = 42  # Use the same random seed for reproducibility
your_outputs = mine.A2_A0273761Y(N)

# Run your friend's code
friend_outputs = friend.A2_A0270412U(N)

# Compare outputs
def compare_outputs(your_output, friend_output, description):
    """
    Compare two numpy arrays or lists and print a message if they are different.
    :param your_output: Output from your code.
    :param friend_output: Output from friend's code.
    :param description: Description of the output being compared.
    """
    # Check if the outputs are numpy arrays
    if isinstance(your_output, np.ndarray) and isinstance(friend_output, np.ndarray):
        # Use np.allclose for floating-point arrays, np.array_equal for exact comparisons
        if np.issubdtype(your_output.dtype, np.floating) or np.issubdtype(friend_output.dtype, np.floating):
            # Attempt to compare transposed versions if shapes mismatch
            if your_output.shape != friend_output.shape:
                if your_output.shape == friend_output.T.shape:
                    if np.allclose(your_output, friend_output.T):
                        print(f"{description} matches perfectly after transpose!")
                    else:
                        print(f"Difference found in {description} after transpose.")
                else:
                    print(f"Shape mismatch in {description}. {your_output.shape} vs {friend_output.shape}.")
            else:
                # Direct comparison if shapes are the same
                if np.allclose(your_output, friend_output):
                    print(f"{description} matches perfectly!")
                else:
                    print(f"Difference found in {description}. Floating-point arrays not close.")
        else:
            if np.array_equal(your_output, friend_output):
                print(f"{description} matches perfectly!")
            else:
                print(f"Difference found in {description}. Arrays not equal.")
    elif isinstance(your_output, list) and isinstance(friend_output, list):
        # Check if lists are the same length
        if len(your_output) != len(friend_output):
            print(f"Length mismatch in {description}. Expected {len(friend_output)} elements but got {len(your_output)}.")
            return
        # Compare each element in the list
        for i, (yo, fo) in enumerate(zip(your_output, friend_output)):
            compare_outputs(yo, fo, f"{description} at index {i}")
    else:
        # For scalar values or other types, use direct comparison
        if np.array_equal(your_output, friend_output):
            print(f"{description} matches perfectly!")
        else:
            print(f"Difference found in {description}. Values differ: {your_output} vs {friend_output}.")

# Compare each output in the required order
compare_outputs(your_outputs[0], friend_outputs[0], "X_train")
compare_outputs(your_outputs[1], friend_outputs[1], "y_train")
compare_outputs(your_outputs[2], friend_outputs[2], "X_test")
compare_outputs(your_outputs[3], friend_outputs[3], "y_test")
compare_outputs(your_outputs[4], friend_outputs[4], "Ytr (One-hot training)")
compare_outputs(your_outputs[5], friend_outputs[5], "Yts (One-hot test)")

# Compare polynomial matrices list
compare_outputs(your_outputs[6], friend_outputs[6], "Ptrain_list")
compare_outputs(your_outputs[7], friend_outputs[7], "Ptest_list")

# Compare weights list
compare_outputs(your_outputs[8], friend_outputs[8], "w_list")

# Compare training and test error arrays
compare_outputs(your_outputs[9], friend_outputs[9], "error_train_array")
compare_outputs(your_outputs[10], friend_outputs[10], "error_test_array")
