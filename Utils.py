import numpy as np
import control
from control import tf, step_response
from scipy.optimize import least_squares
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
    """
    Given a parameter vector [K, theta, tau1, tau2, tau3, tau4],
    build the corresponding 4th-order TF with possible time delay.
    
    G(s) = K * exp(-theta s) / [(1 + tau1*s)(1 + tau2*s)(1 + tau3*s)(1 + tau4*s)]
    """
    K, theta, tau1, tau2, tau3, tau4 = params
    
    # Create s as a symbolic variable for the transfer function
    s = tf('s')
    
    # Denominator = (1 + tau1*s)(1 + tau2*s)(1 + tau3*s)(1 + tau4*s)
    denom = (1 + tau1*s)*(1 + tau2*s)*(1 + tau3*s)*(1 + tau4*s)
    
    # Numerator = K * exp(-theta*s). 
    # However, python-control typically handles time delay with e.g. Pade approximation.
    # We'll handle the 'core' TF first (no delay) and add a Pade approximation for the delay:
    G_nodelay = K / denom
    
    # If you truly need the delay in your TF object, you can approximate it:
    if theta > 1e-8:
        # Pade approximation of e^{-theta*s} (order=1 or 2 for simplicity)
        # Higher-order Pade -> better approximation
        delay_approx = control.pade(theta, n=2)
        num_delay, den_delay = delay_approx
        G_delay = tf(num_delay, den_delay)
        G = G_delay * G_nodelay
    else:
        # If delay is very small, skip or set it to zero
        G = G_nodelay
        
    return G

def simulate_step_response(params, t):
    """
    Simulate the step response of the 4th-order TF defined by `params`
    over the time array `t`.
    
    Returns the output array y_model (same length as t).
    """
    G = build_transfer_function(params)
    
    # Python-control's step_response can take a custom time vector.
    # step_response(G, T=t) returns (y, T_out)
    T_out, y_model = step_response(G, T=t)
    
    return y_model

def objective_function(params, t, y_data):
    """
    Objective function for least-squares fitting:
    
    We want to minimize [y_model(t) - y_data(t)] over time.
    """
    y_model = simulate_step_response(params, t)
    return y_model - y_data

def fit_fourth_order_tf(t, NormalizedY, tankIndex):
    """
    Main function to:
      - Extract y_data for the 4th tank
      - Fit a 4th-order TF
      - Return the best-fit parameters
    """
    # 4th tank is index=3 (zero-based), so:
    y_data = NormalizedY[:, tankIndex]
    
    # Initial guess for parameters [K, theta, tau1, tau2, tau3, tau4].
    # - K ~ 1.0 if truly normalized
    # - theta ~ 0 if we suspect no big delay
    # - tau_i > 0. Let's guess something small or moderate.
    p0 = [1.0, 0.5, 5.0, 5.0, 5.0, 5.0]  # adjust as needed
    
    # Bounds: ensure all tau_i > 0, theta >= 0, K>0 etc.
    # (lower bounds, upper bounds)
    lb = [0.9999, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
    ub = [1.0001, 1e6, 1e6, 1e6, 1e6, 1e6]
    
    # Run least squares
    result = least_squares(
        fun=objective_function,
        x0=p0,
        bounds=(lb, ub),
        args=(t, y_data),
        verbose=1,
        max_nfev=50
    )
    
    best_params = result.x  # [K, theta, tau1, tau2, tau3, tau4]
    return best_params

