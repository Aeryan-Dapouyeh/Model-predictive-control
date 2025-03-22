from ModifiedFourTankSystem import stochasticModifiedFourtankSystem
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Utils import deterministicModifiedFourTankSimulation, stochasticModifiedFourTankSimulation, normal_steady_state, fit_fourth_order_tf
from Utils import simulate_step_response, c2d_zoh, compute_markov_parameters

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

x1, x2, x3, x4, u1, u2, d1, d2 = sp.symbols('x1 x2 x3 x4 u1 u2 d1 d2')

x = sp.Matrix([x1, x2, x3, x4])
u = sp.Matrix([u1, u2])
d = sp.Matrix([d1, d2])

symbolic_Xdot = FTS.symbolicXdot(x, u, d)
symbolic_y = FTS.symbolic_y(x)

Ass, Bss, Bdss, Css, Czss, Dss, Dzss = FTS.ss_matrices(x, u, d, xs, us, ds)



Ass = np.array(Ass).astype(float)
Bss = np.array(Bss).astype(float)
Css = np.array(Css).astype(float)
Dss = np.array(Dss).astype(float)


N=1000
stepChangeIndex=400


################# Q4.6 #################

Ad, Bd, Cd, Dd = c2d_zoh(Ass, Bss, Css, Dss, 1)
# Compute Markov parameters up to step N
H = compute_markov_parameters(Ad, Bd, Cd, Dd, N)


# Plot the markov parameters
for i in range(2):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    axs[0].plot(H[:, 0, i], label=f'y1->u{i}')
    # axs[0].plot(NormalizedY[stepChangeIndex:, 0], label='Normalized y1')
    axs[0].set_xlabel("Timesteps(dt=1)")
    axs[0].legend()

    axs[1].plot(H[:, 1, i], label=f'y2->u{i}')
    # axs[1].plot(NormalizedY[stepChangeIndex:, 1], label='Normalized y2')
    axs[1].set_xlabel("Timesteps(dt=1)")   
    axs[1].legend()

    axs[2].plot(H[:, 2, i], label=f'y3->u{i}')
    # axs[2].plot(NormalizedY[stepChangeIndex:, 2], label='Normalized y3')
    axs[2].set_xlabel("Timesteps(dt=1)") 
    axs[2].legend()

    axs[3].plot(H[:, 3, i], label=f'y4->u{i}')
    # axs[3].plot(NormalizedY[stepChangeIndex:, 3], label='Normalized y4')
    axs[3].set_xlabel("Timesteps(dt=1)")   
    axs[3].legend()

    # Add a shared title
    plt.suptitle(f"Markov parameters for u{i} and y", fontsize=16)

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    # plt.savefig(f"Q4_6_MarkovParameter_u{i}.png", dpi=300, bbox_inches='tight')
    # plt.show()



x0=xs


################# Q4.1 #################


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

    # x_hist_frac, y_hist_frac, z_hist_frac = deterministicModifiedFourTankSimulation(u_frac, x0, N)
    x_hist_frac, y_hist_frac, z_hist_frac, d_hist_frac = stochasticModifiedFourTankSimulation(u=u_frac, x0=x0, N=N, disturbanceStrength=10, R_vv_strength=0.01, noiseType="White_Gaussian")

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
    # plt.savefig(f"Q4_1_StepResponse{fraction}_deterministic.png", dpi=300, bbox_inches='tight')


    # plt.tight_layout()
    # plt.show()


    ################# Q4.3 #################

    NormalizedY = normal_steady_state(y_hist_frac, u_frac, ys, us, stepChangeIndex)
    # from MFTS import normal_steady_state
    # NormalizedY = normal_steady_state(y_hist_frac, u_frac, ys, us, stepChangeIndex, 0)
    # NormalizedY = NormalizedY.T

    plt.figure(figsize=(10, 6))

    plt.plot(NormalizedY[:, 0], label='y1-normal')
    plt.plot(NormalizedY[:, 1], label='y2-normal')
    # plt.plot(NormalizedY[:, 2], label='y3-normal')
    # plt.plot(NormalizedY[:, 3], label='y4-normal')
    plt.legend()

    plt.suptitle(f"Normalized outputs for {fraction*100}% steps with Low disturbance", fontsize=16)
    # plt.savefig(f"Q4_3_StepResponse{fraction}_Normalized_LowNoise.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    # plt.show()        


    ################# Q4.4 #################

    t = np.array([i for i in range(N)])

    best_params_1 = fit_fourth_order_tf(t[stepChangeIndex:], NormalizedY[stepChangeIndex:, :], tankIndex=0)
    best_params_2 = fit_fourth_order_tf(t[stepChangeIndex:], NormalizedY[stepChangeIndex:, :], tankIndex=1)
    best_params_3 = fit_fourth_order_tf(t[stepChangeIndex:], NormalizedY[stepChangeIndex:, :], tankIndex=2)
    best_params_4 = fit_fourth_order_tf(t[stepChangeIndex:], NormalizedY[stepChangeIndex:, :], tankIndex=3)

    best_params_1[0] = 1
    best_params_1[1] = 1
    best_params_1[2] = 1
    best_params_1[3] = 1

    y_1 = simulate_step_response(best_params_1, t[stepChangeIndex:])
    y_2 = simulate_step_response(best_params_2, t[stepChangeIndex:])
    y_3 = simulate_step_response(best_params_3, t[stepChangeIndex:])
    y_4 = simulate_step_response(best_params_4, t[stepChangeIndex:])


    '''
    plt.figure(figsize=(10, 6))

    plt.plot(y_1, label='Fitted model - y_1')
    plt.plot(NormalizedY[stepChangeIndex:, 0], label='Normalized y1')
    plt.plot(y_2, label='Fitted model - y_2')
    plt.plot(NormalizedY[stepChangeIndex:, 1], label='Normalized y2')
    plt.plot(y_3, label='Fitted model - y_3')
    plt.plot(NormalizedY[stepChangeIndex:, 2], label='Normalized y3')
    plt.plot(y_4, label='Fitted model - y_4')
    plt.plot(NormalizedY[stepChangeIndex:, 3], label='Normalized y4')
    plt.legend()

    plt.suptitle(f"Overview of the system for {fraction*100}% steps", fontsize=16)
    # plt.savefig(f"Q4_1_StepResponse{fraction}_NoiseLevel_{noiseLevel}.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    '''

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # States
    axs[0].plot(y_1, label='Fitted model - y1')
    axs[0].plot(NormalizedY[stepChangeIndex:, 0], label='Normalized y1')
    axs[0].set_xlabel("Timesteps(dt=1)")
    axs[0].legend()

    # Noise
    axs[1].plot(y_2, label='Fitted model - y2')
    axs[1].plot(NormalizedY[stepChangeIndex:, 1], label='Normalized y2')
    axs[1].set_xlabel("Timesteps(dt=1)")   
    axs[1].legend()

    # Controller output
    axs[2].plot(y_3, label='Fitted model - y3')
    axs[2].plot(NormalizedY[stepChangeIndex:, 2], label='Normalized y3')
    axs[2].set_xlabel("Timesteps(dt=1)") 
    axs[2].legend()

    # Controller output
    axs[3].plot(y_4, label='Fitted model - y4')
    axs[3].plot(NormalizedY[stepChangeIndex:, 3], label='Normalized y4')
    axs[3].set_xlabel("Timesteps(dt=1)")   
    axs[3].legend()

    # Add a shared title
    plt.suptitle(f"True y versus y obtained from the transfer function {fraction*100}% steps.", fontsize=16)

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(f"Q4_4_StepResponse{fraction}_TransferFunc_LowNoise.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(best_params_1)


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
