from ModifiedFourTankSystem import stochasticModifiedFourtankSystem
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Utils import deterministicModifiedFourTankSimulation, stochasticModifiedFourTankSimulation, normal_steady_state, fit_fourth_order_tf
from Utils import simulate_step_response

np.random.seed(42)

gamma1 = 0.6 
gamma2 = 0.7
# Pipe cross sectional area(cm^2)
a = 2
# Tank cross sectional area(cm^2)
A = 500
g = 982
rho = 1


F1=100
F2=200
F3=200
F4=200
us = np.array([F1, F2])
ds = np.array([0, 0, F3, F4])
xs0 = np.ones(4)*1000

p = [a, a, a, a, A, A, A, A, gamma1, gamma2, g, rho]
R_vv = np.eye(4)*0.1
FTS = stochasticModifiedFourtankSystem(p, xs0, R_vv, noiseType="Deterministic")  


y_prev = []
z_prev = []
x_prev = []
u_prev = []
dk_prev = []


xs = fsolve(lambda xs: FTS.xdot(x=xs, u=us, d=ds, dt=1), xs0)

ys = FTS.y(xs)
zs = FTS.z(xs)


x0=xs


################# Q4.1 #################
N=1000
stepChangeIndex=400

stepFractions = [0.1, 0.25, 0.5]


'''
u_10_F1 = np.ones(N)*F1
u_10_F1[stepChangeIndex:] = F1*(1 + 0.1)
u_10_F2 = np.ones(N)*F2
u_10_F2[stepChangeIndex:] = F2*(1 + 0.1)
u_10 = np.array([u_10_F1, u_10_F2])
x_hist_10, y_hist_10, z_hist_10 = deterministicModifiedFourTankSimulation(u_10, x0, N)
'''

for i, fraction in enumerate(stepFractions):
    u_frac_F1 = np.ones(N)*F1
    u_frac_F1[stepChangeIndex:] = F1*(1 + fraction)

    u_frac_F2 = np.ones(N)*F2
    u_frac_F2[stepChangeIndex:] = F2*(1 + fraction)

    u_frac = np.array([u_frac_F1, u_frac_F2])

    x_hist_frac, y_hist_frac, z_hist_frac = deterministicModifiedFourTankSimulation(u_frac, x0, N)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # States
    axs[0].plot(x_hist_frac[:, 0], label='x1')
    axs[0].plot(x_hist_frac[:, 1], label='x2')
    axs[0].plot(x_hist_frac[:, 2], label='x3')
    axs[0].plot(x_hist_frac[:, 3], label='x4')
    axs[0].set_title("States")
    axs[0].set_xlabel("Timesteps(dt=1)")
    axs[0].set_ylabel("Mass(g)")
    axs[0].legend()

    # Noise
    axs[1].plot(y_hist_frac[:, 0], label='h1-sensor')
    axs[1].plot(y_hist_frac[:, 1], label='h2-sensor')
    axs[1].set_title("Measurments")
    axs[1].set_xlabel("Timesteps(dt=1)")       
    axs[1].set_ylabel("cm")   
    axs[1].legend()

    # Controller output
    axs[2].plot(u_frac[0, :], label='F1')
    axs[2].plot(u_frac[1, :], label='F2')
    axs[2].set_title("Manipulated variables")
    axs[2].set_xlabel("Timesteps(dt=1)")       
    axs[2].set_ylabel("cm3/s")  
    axs[2].legend()

    plt.suptitle(f"Overview of the system for {fraction*100}% steps", fontsize=16)
    # plt.savefig(f"Q3_1_StepResponse{fraction}.png", dpi=300, bbox_inches='tight')


    plt.tight_layout()
    # plt.show()


    ################# Q4.3 #################

    NormalizedY = normal_steady_state(y_hist_frac, u_frac, ys, us, stepChangeIndex)

    plt.figure(figsize=(10, 6))

    plt.plot(NormalizedY[:, 0], label='y1-normal')
    plt.plot(NormalizedY[:, 1], label='y2-normal')
    plt.plot(NormalizedY[:, 2], label='y3-normal')
    plt.plot(NormalizedY[:, 3], label='y4-normal')
    plt.legend()

    plt.suptitle(f"Overview of the system for {fraction*100}% steps", fontsize=16)
    # plt.savefig(f"Q4_3_StepResponse{fraction}_Normalized.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    # plt.show()        


    ################# Q4.4 #################

    t = np.array([i for i in range(N)])

    best_params = fit_fourth_order_tf(t, NormalizedY, tankIndex=3)
    best_params[0] = 1
    y = simulate_step_response(best_params, t)

    plt.figure(figsize=(10, 6))

    plt.plot(y, label='y1-normal')
    plt.plot(NormalizedY[:, 3], label='y2-normal')
    plt.legend()

    plt.suptitle(f"Overview of the system for {fraction*100}% steps", fontsize=16)
    # plt.savefig(f"Q4_1_StepResponse{fraction}_NoiseLevel_{noiseLevel}.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    # plt.show()



    print(best_params)


################# Q4.2 #################

N=1000
stepChangeIndex=400

stepFractions = [0.1, 0.25, 0.5]
noiseLevels = [1, 10, 100]

for i, fraction in enumerate(stepFractions):
    for j, noiseLevel in enumerate(noiseLevels):
        u_frac_F1 = np.ones(N)*F1
        u_frac_F1[stepChangeIndex:] = F1*(1 + fraction)

        u_frac_F2 = np.ones(N)*F2
        u_frac_F2[stepChangeIndex:] = F2*(1 + fraction)

        u_frac = np.array([u_frac_F1, u_frac_F2])

        x_hist_frac, y_hist_frac, z_hist_frac, d_hist_frac = stochasticModifiedFourTankSimulation(u=u_frac, x0=x0, N=N, disturbanceStrength=noiseLevel, R_vv_strength=0.1, noiseType="White_Gaussian")

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # States
        axs[0].plot(x_hist_frac[:, 0], label='x1')
        axs[0].plot(x_hist_frac[:, 1], label='x2')
        axs[0].plot(x_hist_frac[:, 2], label='x3')
        axs[0].plot(x_hist_frac[:, 3], label='x4')
        axs[0].set_title("States")
        axs[0].set_xlabel("Timesteps(dt=1)")
        axs[0].set_ylabel("Mass(g)")
        axs[0].legend()

        # Noise
        axs[1].plot(y_hist_frac[:, 0], label='h1-sensor')
        axs[1].plot(y_hist_frac[:, 1], label='h2-sensor')
        axs[1].set_title("Measurments")
        axs[1].set_xlabel("Timesteps(dt=1)")       
        axs[1].set_ylabel("cm")   
        axs[1].legend()

        # Controller output
        axs[2].plot(u_frac[0, :], label='F1')
        axs[2].plot(u_frac[1, :], label='F2')
        axs[2].set_title("Manipulated variables")
        axs[2].set_xlabel("Timesteps(dt=1)")       
        axs[2].set_ylabel("cm3/s")  
        axs[2].legend()

        plt.suptitle(f"Overview of the system for {fraction*100}% steps - Noise level: {noiseLevel}", fontsize=16)
        # plt.savefig(f"Q4_2_StepResponse{fraction}_NoiseLevel_{noiseLevel}.png", dpi=300, bbox_inches='tight')

        plt.tight_layout()
        # plt.show()
