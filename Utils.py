import numpy as np
import control
from control import tf, step_response
from scipy.optimize import least_squares
from scipy.linalg import expm
from ModifiedFourTankSystem import stochasticModifiedFourtankSystem


def deterministicModifiedFourTankSimulation(u, x0, N):
    gamma1 = 0.6 
    gamma2 = 0.7
    a = 2
    A = 500
    g = 982
    rho = 1

    p = [a, a, a, a, A, A, A, A, gamma1, gamma2, g, rho]
    R_vv = np.eye(4)*0.1
    FTS = stochasticModifiedFourtankSystem(p, x0, R_vv, noiseType="Deterministic")  
    
    x_prev = [x0]
    y_prev = []
    z_prev = []

    xk = x0

    for i in range(N):
        dk = np.zeros(4)
        uk = u[:, i]

        xdot = FTS.xdot(xk, uk, dk, dt=1)
        yk = FTS.y(xk)
        zk = FTS.z(xk)

        y_prev.append(yk)
        z_prev.append(zk)

        xk = np.add(xk, xdot)
        x_prev.append(xk)
    
    x_prev = np.array(x_prev)
    y_prev = np.array(y_prev)
    z_prev = np.array(z_prev)

    return x_prev, y_prev, z_prev



def stochasticModifiedFourTankSimulation(u, x0, N, disturbanceStrength=1, R_vv_strength=0.1, noiseType="Weiner_process"):
    gamma1 = 0.6 
    gamma2 = 0.7
    a = 2
    A = 500
    g = 982
    rho = 1

    p = [a, a, a, a, A, A, A, A, gamma1, gamma2, g, rho]
    R_vv = np.eye(4)*R_vv_strength
    FTS = stochasticModifiedFourtankSystem(p, x0, R_vv, noiseType=noiseType)  
    
    x_prev = np.matrix(x0)
    y_prev = []
    z_prev = []
    d_prev = []

    xk = x0

    for i in range(N):
        dk = FTS.d(dt=disturbanceStrength)
        d_prev.append(dk)

        uk = u[:, i]

        xdot = FTS.xdot(xk, uk, dk, dt=1)
        yk = FTS.y(xk)
        zk = FTS.z(xk)

        y_prev.append(yk)
        z_prev.append(zk)

        xk = np.add(xk, xdot)
        xk = np.maximum(xk, 0)
        xk_toAdd = xk.reshape(1, 4)
        x_prev = np.concatenate((x_prev, xk_toAdd), axis=0)


    x_prev = np.array(x_prev)
    y_prev = np.array(y_prev)
    z_prev = np.array(z_prev)
    d_prev = np.array(d_prev)

    return x_prev, y_prev, z_prev, d_prev




def normal_steady_state(y, u, ys, us, index_step):

    T, tankNumber = y.shape

    normalY = np.zeros((T, tankNumber))

    for i in range(index_step, T):
        normalY[i] = np.divide((y[i, :] - y[index_step]), (y[T-1] - y[index_step]))
        normalY[:index_step, :] = np.zeros(4)

    return normalY



def build_transfer_function(params):

    K, theta, tau1, tau2, tau3, tau4 = params
    s = tf('s')
    denom = (1 + tau1*s)*(1 + tau2*s)*(1 + tau3*s)*(1 + tau4*s)
    G_nodelay = K / denom
    if theta > 1e-8:
        delay_approx = control.pade(theta, n=2)
        num_delay, den_delay = delay_approx
        G_delay = tf(num_delay, den_delay)
        G = G_delay * G_nodelay
    else:
        G = G_nodelay
        
    return G

def simulate_step_response(params, t):
    G = build_transfer_function(params)
    T_out, y_model = step_response(G, T=t)
    
    return y_model

def objective_function(params, t, y_data):
    y_model = simulate_step_response(params, t)
    return y_model - y_data

def fit_fourth_order_tf(t, NormalizedY, tankIndex):

    y_data = NormalizedY[:, tankIndex]
    
    p0 = [1.0, 0.5, 5.0, 5.0, 5.0, 5.0]  

    lb = [0.9999, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
    ub = [1.0001, 1e6, 1e6, 1e6, 1e6, 1e6]

    result = least_squares(
        fun=objective_function,
        x0=p0,
        bounds=(lb, ub),
        args=(t, y_data),
        verbose=1,
        max_nfev=50
    )
    
    best_params = result.x 
    return best_params

def c2d_zoh(A, B, C, D, Ts):

    Ad = expm(A * Ts)

    I = np.eye(A.shape[0])

    A_inv = np.linalg.inv(A)
    Bd = A_inv @ (Ad - I) @ B
    
    Cd = C.copy()
    Dd = D.copy()

    return Ad, Bd, Cd, Dd

def compute_markov_parameters(Ad, Bd, Cd, Dd, N):

    H = [Dd]  
    A_power = np.eye(Ad.shape[0]) 

    for k in range(1, N+1):
        A_power = A_power @ Ad 
        H.append(Cd @ A_power @ Bd)

    H = np.array(H)
    
    return H

