from scipy.linalg import expm
import numpy as np

def c2dzoh(A, B, Ts):

    nx, nu = B.shape
    M = np.block([
        [A, B],
        [np.zeros((nu, nx)), np.zeros((nu, nu))]
    ])
    Phi = expm(M * Ts)

    Abar = Phi[:nx, :nx]
    Bbar = Phi[:nx, nx:nx + nu]

    return Abar, Bbar