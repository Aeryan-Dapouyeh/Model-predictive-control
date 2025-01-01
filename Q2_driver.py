from ModifiedFourTankSystem import stochasticModifiedFourtankSystem

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

gamma1 = 0.6 
gamma2 = 0.7
# Pipe cross sectional area(cm^2)
a = 2
# Tank cross sectional area(cm^2)
A = 500
g = 982
rho = 1

timeSteps = 1000

# Initialize the four tank system
x0 = np.array([2.0, 2.0, 2.0, 2.0])
u0 = np.ones((2, 1))
p = [a, a, a, a, A, A, A, A, gamma1, gamma2, g, rho]
R_vv = np.eye(4)
FTS = stochasticModifiedFourtankSystem(p, x0, R_vv, noiseType="Weiner_process")   # noisetype could be: Deterministic, White_Gaussian or Weiner_process

# List of values at each timestep
y_prev = []
z_prev = []
x_prev = np.matrix(x0)
u_prev = np.matrix(u0)
dk_prev = []

xk = x0
uk = u0

for i in range(timeSteps):
    dk = FTS.d()
    dk_prev.append(dk)

    xdot = FTS.xdot(xk, uk, dk)
    xk += xdot
    xk = np.maximum(xk, 0)
    xk_toAdd = xk.reshape(1, 4)
    x_prev = np.concatenate((x_prev, xk_toAdd), axis=0)
    # x_prev = np.append(x_prev, xk, axis=0)
    # x_prev = x_prev.append(xk)

    y_k = FTS.y(xk)
    y_prev.append(y_k)

    z_k = FTS.z(xk)
    z_prev.append(z_k)


z_prev = np.array(z_prev)
dk_prev = np.array(dk_prev)
y_prev = np.array(y_prev)


fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# States
axs[0].plot(x_prev[:, 0], label='x1')
axs[0].plot(x_prev[:, 1], label='x2')
axs[0].plot(x_prev[:, 2], label='x3')
axs[0].plot(x_prev[:, 3], label='x4')
axs[0].set_title("States")
axs[0].set_xlabel("Timesteps(dt=1)")
axs[0].set_ylabel("Mass(g)")
axs[0].legend()

# Noise
axs[1].plot(dk_prev[:, 2], label='F3')
axs[1].plot(dk_prev[:, 3], label='F4')
axs[1].set_title("Noise")
axs[1].set_xlabel("Timesteps(dt=1)")       
axs[1].set_ylabel("(cm3/s)")   
axs[1].legend()

# Controller output
axs[2].plot(z_prev[:, 0], label='z1')
axs[2].plot(z_prev[:, 1], label='z2')
axs[2].set_title("Controller output")
axs[2].set_xlabel("Timesteps(dt=1)")       
axs[2].set_ylabel("Levels(cm)")  
axs[2].legend()


# plt.savefig("Q2_4_Weiner_process.png", dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.show()


