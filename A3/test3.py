import numpy as np
import A3_A0277148U as mine
import A3_A0252168A as friend

def test(learning_rate, num_iters):
    a1_out, f1_1_out, b1_out, f2_1_out, c1_out, d1_out, f3_1_out = mine.A3_A0277148U(learning_rate, num_iters)
    a2_out, f1_2_out, b2_out, f2_2_out, c2_out, d2_out, f3_2_out = friend.A3_A0252168A(learning_rate, num_iters)

    # Compare outputs for both functions from file1 and file2
    assert np.allclose(a1_out, a2_out), "a_out values do not match!"
    assert np.allclose(f1_1_out, f1_2_out), "f1_out values do not match!"
    assert np.allclose(b1_out, b2_out), "b_out values do not match!"
    assert np.allclose(f2_1_out, f2_2_out), "f2_out values do not match!"
    assert np.allclose(c1_out, c2_out), "c_out values do not match!"
    assert np.allclose(d1_out, d2_out), "d_out values do not match!"
    assert np.allclose(f3_1_out, f3_2_out), "f3_out values do not match!"

    print("All outputs match for the given learning rate and number of iterations!")

if __name__ == "__main__":
    learning_rate = 0.01
    num_iters = 10
    test(learning_rate, num_iters)
