import numpy as np
from QPSolver import qpsolver
from qpsolvers import solve_qp


def LMPCcompute_inputcons_outputSoftCons(R, X0, D, U, Ad, Bd, Bd_d, Css, u_min, u_max, u_delta_min, u_delta_max, z_min, z_max, N, Q_cof, u_delta_cof, eta_cof1, eta_cof2):
    num_x = Ad.shape[0]
    num_u = Bd.shape[1]
    num_d = Bd_d.shape[1]
    num_y = 4
    num_z = 2
    num_eta = num_y
    num_u_delta_cons = N * num_u
    num_z_cons = N * num_y

    Phi = np.zeros((N * num_y, num_x))
    Phi_d = np.zeros((N * num_y, num_d))
    Gam = np.zeros((N * num_y, N * num_u + 2 * N * num_eta))

    Qz = np.diag(np.zeros(N * num_y))
    Sz = np.diag(u_delta_cof * np.ones(num_u))

    Hs = np.zeros((N * num_u + 2 * N * num_eta, N * num_u + 2 * N * num_eta))
    Mu_delta = np.zeros((N * num_u + 2 * N * num_eta, num_u))
    Mu_delta[:num_u, :] = -Sz

    H_eta1 = np.diag(np.concatenate([np.zeros(N * num_u), eta_cof1 * np.ones(N * num_eta), np.zeros(N * num_eta)]))
    H_eta2 = np.diag(np.concatenate([np.zeros(N * num_u), np.zeros(N * num_eta), eta_cof2 * np.ones(N * num_eta)]))

    U0 = np.zeros((N * num_u + 2 * N * num_eta, 1))
    Iu = np.eye(num_u)
    Au_delta_cons = np.zeros((num_u_delta_cons, N * num_u + 2 * N * num_eta))

    for i in range(1, N + 1):
        Phi[(i - 1) * num_y:i * num_y, :] = np.linalg.matrix_power(Ad, i) @ Css
        Phi_d[(i - 1) * num_y:i * num_y, :] = np.linalg.matrix_power(Ad, i) @ Bd_d

        Gam_i = np.zeros((num_y, N * num_u + 2 * N * num_eta))
        for j in range(1, i + 1):
            Gam_i[:, (j - 1) * num_u:j * num_u] = Css @ np.linalg.matrix_power(Ad, i - j) @ Bd
        Gam[(i - 1) * num_y:i * num_y, :] = Gam_i

        qz = np.diag(np.concatenate([Q_cof * np.ones(num_z), np.zeros(num_y - num_z)]))
        Qz[(i - 1) * num_y:i * num_y, (i - 1) * num_y:i * num_y] = qz

        if i == 1:
            Hs[:num_u, :2 * num_u] = np.block([[2 * Sz, -Sz]])
        elif i == N:
            Hs[(i - 1) * num_u:i * num_u, (i - 2) * num_u:i * num_u] = np.block([[-Sz, Sz]])
        else:
            Hs[(i - 1) * num_u:i * num_u, (i - 2) * num_u:(i + 1) * num_u] = np.block([[-Sz, 2 * Sz, -Sz]])

        U0[(i - 1) * num_u:i * num_u, :] = U.reshape(-1, 1)

        if i == 1:
            Au_delta_cons[:num_u, :num_u] = Iu
        else:
            Au_delta_cons[(i - 1) * num_u:i * num_u, (i - 2) * num_u:i * num_u] = np.block([[-Iu, Iu]])

    Mx0 = Gam.T @ Qz @ Phi
    Md = Gam.T @ Qz @ Phi_d
    Mr = -Gam.T @ Qz
    Hr = Gam.T @ Qz @ Gam

    if(len(U.shape)==1):
        U = np.array([U]).T

    Hu = Hr + Hs + H_eta1 + H_eta2
    g = Mx0 @ X0 + Mr @ R + Mu_delta @ U + Md @ D

    Az_cons_min = Gam.copy()
    Az_cons_min[:, N * num_u:N * num_u + N * num_eta] = np.diag(np.ones(N * num_eta))
    Az_cons_min[:, N * num_u + N * num_eta:] = np.diag(np.zeros(N * num_eta))

    Az_cons_max = Gam.copy()
    Az_cons_max[:, N * num_u:N * num_u + N * num_eta] = np.diag(np.zeros(N * num_eta))
    Az_cons_max[:, N * num_u + N * num_eta:] = np.diag(-np.ones(N * num_eta))

    Z_min = np.tile(z_min.reshape(-1, 1), (N, 1)) - Phi @ X0 - Phi_d @ D
    Z_max = np.tile(z_max.reshape(-1, 1), (N, 1)) - Phi @ X0 - Phi_d @ D

    U_min = np.concatenate([u_min * np.ones(N * num_u), np.zeros(2 * N * num_eta)])
    U_min = np.array([U_min]).T
    U_max = np.concatenate([u_max * np.ones(N * num_u), 5000 * np.ones(2 * N * num_eta)])
    U_max = np.array([U_max]).T

    U_delta_min = np.array([np.full(num_u_delta_cons, u_delta_min)]).T
    U_delta_max = np.array([np.full(num_u_delta_cons, u_delta_max)]).T

    U_delta_min[:num_u] += U[:num_u]
    U_delta_max[:num_u] += U[:num_u]

    A_ieq = np.vstack([Au_delta_cons, Az_cons_min, Az_cons_max])
    b_ieq_min = np.vstack((U_delta_min, Z_min, np.zeros((num_z_cons, 1))))
    b_ieq_max = np.vstack((U_delta_max, (5000 * np.ones((num_z_cons, 1))), Z_max))



    u_opt, _ = qpsolver(0.5 * (Hu + Hu.T), g, U_min, U_max, A_ieq, b_ieq_min, b_ieq_max, U0)
    return u_opt[:num_u]



