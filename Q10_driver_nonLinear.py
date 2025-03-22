import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from c2dzoh import c2dzoh
from control import dlqe
from MPCInputConSoftOutputCon import MPCInputConSoftOutputCon
from ModifiedFourTankSystem import stochasticModifiedFourtankSystem

np.random.seed(42)

# System parameters
a = 2
A = 500
gamma1 = 0.6
gamma2 = 0.7 
g = 981  
rho = 1.00  

p = [a, a, a, a, A, A, A, A, gamma1, gamma2, g, rho]

# Steady State
F1 = 100
F2 = 200
F3 = 200
F4 = 200

us = np.array([F1, F2])
ds = np.array([0, 0, F3, F4])

xs0 = np.ones(4)*1000


R_vv = np.eye(4)*1
FTS = stochasticModifiedFourtankSystem(p, xs0, R_vv, noiseType="White_Gaussian")  

xs = fsolve(lambda xs: FTS.xdot(x=xs, u=us, d=ds, dt=1), xs0)


ys = FTS.y(xs)
zs = FTS.z(xs)

x1, x2, x3, x4, u1, u2, d1, d2 = sp.symbols('x1 x2 x3 x4 u1 u2 d1 d2')

x = sp.Matrix([x1, x2, x3, x4])
u = sp.Matrix([u1, u2])
d = sp.Matrix([d1, d2])

symbolic_Xdot = FTS.symbolicXdot(x, u, d)
symbolic_y = FTS.symbolic_y(x)

Ass, Bss, Bdss, Css, Czss, Dss, Dzss = FTS.ss_matrices(x, u, d, xs, us, ds)

Ass = np.array(Ass).astype(float)
Bss = np.array(Bss).astype(float)
Bdss = np.array(Bdss).astype(float)
Css = np.array(Css).astype(float)
Czss = np.array(Czss).astype(float)
Dss = np.array(Dss).astype(float)
Dzss = np.array(Dzss).astype(float)

Ts = 4
Ad, Bd = c2dzoh(Ass, Bss, Ts)
Ad, Bd_d = c2dzoh(Ass, Bdss, Ts)
Gw_d = Bd_d
ss_matrices = (Ad, Bd, Bd_d, Czss)


Ad_kalman = np.block([
    [Ad, Gw_d],
    [np.zeros((2, 4)), np.eye(2)]
])

Bd_kalman = np.vstack([Bd, np.zeros((2, 2))])
Cd_kalman = np.hstack([np.array(Czss.tolist(), dtype=float), np.zeros((2, 2))])
Cdy_kalman = np.hstack([np.array(Css.tolist(), dtype=float), np.zeros((4, 2))])

W_kalman = np.diag([1, 1, 1, 1, 0.001, 0.001]) * Ts
V_kalman = np.diag([1, 1])
Vy_kalman = np.diag([1, 1, 1, 1])
G_w = np.block([
    [np.diag([1, 1, 1, 1]), Gw_d],
    [np.zeros((2, 4)), np.eye(2)]
])


L_kalman, P_kalman, E_kalman = dlqe(Ad_kalman,G_w,Cdy_kalman,W_kalman,Vy_kalman)




# The input constrained MPC with soft output constraints experiment starts...

# -------------------------------------------------------------------------
# Initialization of parameters
# -------------------------------------------------------------------------
t0 = 0.0          # Initial time
t_final = 80 * 40 # Final time
t = np.arange(t0, t_final + Ts, Ts)
N = len(t)

x0 = xs
# or for some situations: 
# x0 = np.zeros((4, 1))  

# -------------------------------------------------------------------------
# Process Noise
# -------------------------------------------------------------------------
Q = np.array([[2.0**2,      0.0],
              [0.0,    2.0**2]])
Lq = np.linalg.cholesky(Q)  
w = Lq @ np.random.randn(2, N)

# -------------------------------------------------------------------------
# Measurement Noise
# -------------------------------------------------------------------------
R_low = np.eye(4)
Lr_low = np.linalg.cholesky(R_low)
v_low = Lr_low @ np.random.randn(4, N)

R_mid = 10.0 * np.eye(4)
Lr_mid = np.linalg.cholesky(R_mid)
v_mid = Lr_mid @ np.random.randn(4, N)

R_high = 20.0 * np.eye(4)
Lr_high = np.linalg.cholesky(R_high)
v_high = Lr_high @ np.random.randn(4, N)

# -------------------------------------------------------------------------
# Disturbance
# -------------------------------------------------------------------------
disturbance_step = 100

d = np.vstack((50.0 * np.ones(N), 50.0 * np.ones(N)))  
d[0, disturbance_step:] = 100.0

dw = d + w 

# Define constants
xdim = 4
udim = 2
ddim = 2
ydim = 4
zdim = 2
num_eta = ydim
zbar = 30.0
N_horizon = 50
u_min = 0.0
u_max = 300.0
u_delta_min = -40.0
u_delta_max = 40.0

z_min = np.array([-1000, -1000, -1000, -1000])
z_min_error = np.array([zbar - 10, zbar - 10, -1000, -1000])
z_min_safe = np.array([0, 0, 0, 0])
z_max = np.array([zbar + 10, zbar + 10, zbar + 1000, zbar + 1000])

# Initialize matrices
N = 500
x = np.zeros((xdim, N))
y = np.zeros((ydim, N))
z = np.zeros((zdim, N))
u = np.zeros((udim, N))
x_bar = np.zeros((xdim, N))
d_bar = np.zeros((ddim, N))
y_bar = np.zeros((ydim, N))

d0 = ds[2:]
y0 = ys

x[:, 0] = x0.flatten()
x_bar[:, 0] = x0.flatten()
d_bar[:, 0] = d0.flatten()
y_bar[:, 0] = y0.flatten()
xk = x_bar[:, 0]

rk = zbar * np.ones((N_horizon * ydim, 1))
dk = d_bar[:, 0]
uk = np.zeros((udim, 1))
us = np.zeros((udim,))
sk = np.tile(us[0], (N_horizon * udim, 1))

Q_cof = 0.03
u_delta_cof = 10
eta_cof1 = 10
eta_cof2 = 10
z_flag = 0

# Simulation loop
for k in range(N - 1):
    # ---------------------------------------------------------------------
    # 1) Sensor feedback
    _xk = xk.reshape(-1, 1)
    _uk = uk.reshape(-1, 1)
    # yk = Css @ _xk + Dss @ _uk + v_low[:, [k]]
    yk = FTS.y(xk)
    yk = np.array([yk]).T
    yk_no_noise = Css @ _xk + Dss @ _uk
    # zk = yk_no_noise[0:zdim, :] 
    zk = FTS.z(xk)
    zk = np.array([zk]).T

    # ---------------------------------------------------------------------
    # 2) Kalman observer update (Static Kalman example)
    # ---------------------------------------------------------------------
    x_kalman = np.vstack((x_bar[:, [k]], d_bar[:, [k]])) 

    tmp = yk - (Cdy_kalman @ x_kalman + Dss @ _uk)  # innovation
    x_kalman = Ad_kalman @ x_kalman + Bd_kalman @ _uk + (Ad_kalman @ L_kalman) @ tmp

    Y_bar = Cdy_kalman @ x_kalman  # predicted measurement

    # Extract states and disturbance from x_kalman
    xk = x_kalman[0:xdim, :]
    dk = x_kalman[xdim:xdim + ddim, :]

    if yk[1] > (z_min_error[1] - 5):
        z_min = z_min_error
        z_flag = 1
    else:
        z_min = z_min_safe

    if z_flag == 1:
        z_min = z_min_error

    # ---------------------------------------------------------------------
    # 3) MPC controller
    # ---------------------------------------------------------------------
    uk = MPCInputConSoftOutputCon(
        rk, xk, dk, uk, Ad, Bd, Bd_d, Css, u_min, u_max, u_delta_min, u_delta_max, 
        z_min, z_max, N_horizon, Q_cof, u_delta_cof, eta_cof1, eta_cof2
    )
    # State update
    xdot = FTS.xdot(xk, uk, dk)[0]
    xdot = np.array([xdot]).T
    x_next = xk+xdot
    x_k = x_next
    # xk = Ad @ x[:, k] + Bd @ uk + Bd_d @ d[:, k] + Gw_d @ w[:, k]
    x[:, k + 1] = xk.flatten()
    y[:, k] = yk.flatten()
    z[:, k] = zk.flatten()
    u[:, k + 1] = uk.flatten()
    x_bar[:, k + 1] = x_kalman[:xdim, 0]
    d_bar[:, k + 1] = x_kalman[xdim:, 0]
    y_bar[:, k + 1] = Y_bar.flatten()

    print(f"{k}/{N-1}")

# Final step
k = N - 1
yk = FTS.y(xk)
yk = np.array([yk]).T
yk_no_noise = Css @ xk + Dss @ uk

zk = FTS.z(xk)
zk = np.array([zk]).T
y[:, k] = yk.flatten()
z[:, k] = zk.flatten()


fig, axs = plt.subplots(1, 4, figsize=(12, 4))

# States
axs[0].plot(x[0, :], label='x1')
axs[0].plot(x[1, :], label='x2')
axs[0].plot(x[2, :], label='x3')
axs[0].plot(x[3, :], label='x4')
axs[0].set_title("States")
axs[0].set_xlabel("Timesteps(dt=1)")
axs[0].set_ylabel("Mass(g)")
axs[0].legend()

# Noise
axs[1].plot(y[0, :], label='y1')
axs[1].plot(y[1, :], label='y2')
axs[1].plot(y[2, :], label='y3')
axs[1].plot(y[3, :], label='y4')
axs[1].set_title("Measurments")
axs[1].set_xlabel("Timesteps(dt=1)")       
axs[1].set_ylabel("Levels(cm3)")   
axs[1].legend()

# Noise
axs[2].plot(z[0, :], label='z1')
axs[2].plot(z[1, :], label='z2')
axs[2].plot(range(z.shape[1]), [zbar] * z.shape[1], label='z1 bar - set point', linestyle='--', color='blue')
axs[2].plot(range(z.shape[1]), [zbar] * z.shape[1], label='z2 bar - set point', linestyle='--', color='red')
axs[2].set_title("Controlled variables")
axs[2].set_xlabel("Timesteps(dt=1)")       
axs[2].set_ylabel("Levels(cm3)")   
axs[2].legend()


axs[3].plot(u[0, :], label='u1')
axs[3].plot(u[1, :], label='u2')
axs[3].plot(range(z.shape[1]), [u_min] * z.shape[1], label='u_min', linestyle='--', color='blue')
axs[3].plot(range(z.shape[1]), [u_max] * z.shape[1], label='u_max', linestyle='--', color='red')
axs[3].set_title("Manipulated variables")
axs[3].set_xlabel("Timesteps(dt=1)")       
axs[3].set_ylabel("Flow(cm^3)")   
axs[3].legend()


# plt.savefig("Q2_4_Weiner_process.png", dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.show()





