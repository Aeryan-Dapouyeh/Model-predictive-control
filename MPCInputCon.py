import numpy as np
from QPSolver import qpsolver

def InputConstraindMPC(R, X0, U, D, ss_matrices, u_bounds, u_deltaBounds, Q_cof, u_delta_cof, N_horizon):
    
    Ad, Bd, Bd_d, Czss = ss_matrices
    u_min, u_max = u_bounds
    u_delta_min, u_delta_max = u_deltaBounds
    
    xdim = Ad.shape[0]
    udim = Bd.shape[1]
    ddim = Bd_d.shape[1]
    zdim = 2 

    u_delta_con_dim = N_horizon * udim

    Phi = np.zeros((N_horizon * zdim, xdim))
    Phi_d = np.zeros((N_horizon * zdim, ddim))
    Gam = np.zeros((N_horizon * zdim, N_horizon * udim))

    Qz = np.diag(Q_cof * np.ones(N_horizon * zdim))
    Sz = np.diag(u_delta_cof * np.ones(udim))

    Hs = np.zeros((N_horizon * udim, N_horizon * udim))
    Mu_delta = np.zeros((N_horizon * udim, udim))
    Mu_delta[:udim, :] = -Sz

    U0 = np.zeros((N_horizon * udim, 1))
    Iu = np.eye(udim)
    Au_delta_cons = np.zeros((u_delta_con_dim, N_horizon * udim))

    for i in range(1, N_horizon + 1):
        Phi[(i - 1) * zdim:i * zdim, :] = Czss @ np.linalg.matrix_power(Ad, i)
        Phi_d[(i - 1) * zdim:i * zdim, :] = Czss @ np.linalg.matrix_power(Ad, i) @ Bd_d

        Gam_i = np.zeros((zdim, N_horizon * udim))
        for j in range(1, i + 1):
            Gam_i[:, (j - 1) * udim:j * udim] = Czss @ np.linalg.matrix_power(Ad, i - j) @ Bd

        Gam[(i - 1) * zdim:i * zdim, :] = Gam_i

        if i == 1:
            Hs[:udim, :2 * udim] = np.block([[2 * Sz, -Sz]])
        elif i == N_horizon:
            Hs[-udim:, -2 * udim:] = np.block([[-Sz, Sz]])
        else:
            Hs[(i - 1) * udim:i * udim, (i - 2) * udim:(i + 1) * udim] = np.block([[-Sz, 2 * Sz, -Sz]])

        U0[(i - 1) * udim:i * udim, :] = np.array([U]).T # U

        if i == 1:
            Au_delta_cons[:udim, :udim] = Iu
        else:
            Au_delta_cons[(i - 1) * udim:i * udim, (i - 2) * udim:i * udim] = np.block([[-Iu, Iu]])

    Mx0 = Gam.T @ Qz @ Phi
    Md = Gam.T @ Qz @ Phi_d
    Mr = -Gam.T @ Qz
    Hr = Gam.T @ Qz @ Gam

    Hu = Hr + Hs

    if(len(U.shape)==1):
        U = np.array([U]).T

    g = Mx0 @ X0 + Mr @ R + Mu_delta @ U + Md @ D

    U_min = u_min * np.ones((N_horizon * udim, 1))
    U_max = u_max * np.ones((N_horizon * udim, 1))

    U_delta_min = u_delta_min * np.ones((u_delta_con_dim, 1))
    U_delta_max = u_delta_max * np.ones((u_delta_con_dim, 1))

    U_delta_min[0] = u_delta_min + U[0]
    U_delta_min[1] = u_delta_min + U[1]
    U_delta_max[0] = u_delta_max + U[0]
    U_delta_max[1] = u_delta_max + U[1]

    u_opt, _ = qpsolver(0.5 * (Hu + Hu.T), g, U_min, U_max, Au_delta_cons, U_delta_min, U_delta_max, U0)

    u_new = u_opt[:udim]
    return u_new
