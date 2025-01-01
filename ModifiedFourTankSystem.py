import numpy as np
import sympy as sp

class stochasticModifiedFourtankSystem():
    def __init__(self, p, x0, R_vv, noiseType="Weiner_process"):
        self.p = p
        self.a1 = p[0]
        self.a2 = p[1]
        self.a3 = p[2]
        self.a4 = p[3]
        self.A1 = p[4]
        self.A2 = p[5]
        self.A3 = p[6]
        self.A4 = p[7]
        self.gamma1 = p[8]
        self.gamma2 = p[9]
        self.g = p[10]
        self.rho = p[11]

        self.a = np.array([self.a1, self.a2, self.a3, self.a4])
        self.A = np.array([self.A1, self.A2, self.A3, self.A4])

        self.R_vv = R_vv

        self.x0 = x0

        self.prev_d = np.zeros(4)

        self.noiseType = noiseType


    # Disturbance function
    def d(self, sigma=1, dt=1):
        
        if self.noiseType=="Deterministic":
            d = np.zeros(4)
            self.prev_d = d
            return d
        elif self.noiseType=="White_Gaussian":
            d3 = np.random.normal(loc=0, scale=dt)
            d4 = np.random.normal(loc=0, scale=dt)
        else: 
            d3 = np.random.normal(loc=self.prev_d[2], scale=dt)
            d4 = np.random.normal(loc=self.prev_d[3], scale=dt)
        
        d3 = max(0.0, d3)
        d4 = max(0.0, d4)

        d = np.array([
            0, 0, d3, d4
        ])
        
        self.prev_d = d

        return d

    # Dynamics function
    def xdot(self, x, u, d, dt=1):
        
        F1 = u[0]
        F2 = u[1]
        q_in = np.array([
            self.gamma1*F1, 
            self.gamma2*F2, 
            (1 - self.gamma2)*F2, 
            (1 - self.gamma1)*F1
        ])
        q = np.zeros((4, 1))
        for i in range(4): 
            q[i] = self.a[i]*np.sqrt(2.0*self.g*(x[i]/(self.rho*self.A[i])))
        driftTerm = np.array([
            self.rho*(q_in[0] + q[2] - q[0]), 
            self.rho*(q_in[1] + q[3] - q[1]), 
            self.rho*(q_in[2] - q[2]),
            self.rho*(q_in[3] - q[3]) 
        ])
        xdot = np.array(driftTerm[:, 0] + d)

        return xdot

    # Sensor function
    def y(self, x):
        
        y = np.zeros((4, 1))
        for i in range(4):
            y[i] = x[i]/(self.rho*self.A[i])
        
        v = np.random.multivariate_normal(mean=np.zeros(4), cov=self.R_vv)
        v = np.array([max(0, e) for e in v])
        v = np.add(v, np.ones(4)*1e-6)

        if self.noiseType=="Deterministic": 
            return np.array(y[:, 0])
        else:
            return np.array(y[:, 0] + v)

    def z(self, x):
        z = np.zeros(2)
        for i in range(2):
            z[i] = x[i]/(self.rho*self.A[i])
        
        return z
    

    def symbolicXdot(self, x, u, d):
        
        qin = sp.Matrix([
            self.gamma1 * u[0],
            self.gamma2 * u[1],
            (1 - self.gamma2) * u[1],
            (1 - self.gamma1) * u[0]
        ])

        h = sp.Matrix([x[i] / (self.rho * self.A[i]) for i in range(len(x))])  
        h_sqrt = sp.Matrix([sp.sqrt(2 * self.g * h[i]) for i in range(len(h))])
        qout = sp.Matrix([self.a[i] * h_sqrt[i] for i in range(len(self.a))])

        xdot = sp.Matrix([
        self.rho * (qin[0] + qout[2] - qout[0]),  
        self.rho * (qin[1] + qout[3] - qout[1]),  
        self.rho * (qin[2] - qout[2]) + d[0],   
        self.rho * (qin[3] - qout[3]) + d[1]  
        ])

        return xdot
    
    def symbolic_y(self, x):
        return sp.Matrix([x[i] / (self.rho * self.A[i]) for i in range(len(x))])
    
    def symbolic_z(self, x):
        return sp.Matrix([x[i] / (self.rho * self.A[i]) for i in range(2)])

    def ss_matrices(self, x, u, d, xs, us, ds):
        symbolicXdot = self.symbolicXdot(x, u, d)
        y = self.symbolic_y(x)
        F = sp.Matrix([u[0], u[1], d[0], d[1]]) 

        Ass = sp.N(symbolicXdot.jacobian(x).subs({**dict(zip(x, xs)), **dict(zip(F, us + ds[2:]))}), 4)
        Bss = sp.N(symbolicXdot.jacobian(F[:2]).subs({**dict(zip(x, xs)), **dict(zip(F, us + ds[2:]))}), 4)
        Bdss = sp.N(symbolicXdot.jacobian(F[2:]).subs({**dict(zip(x, xs)), **dict(zip(F, us + ds[2:]))}), 4)
        Css = sp.N(y.jacobian(x).subs({**dict(zip(x, xs)), **dict(zip(F, us + ds[2:]))}), 4)
        Czss = Css[:2, :]
        Dss = sp.N(y.jacobian(F[:2]).subs({**dict(zip(x, xs)), **dict(zip(F, us + ds[2:]))}), 4)
        Dzss = Dss[:2, :]

        Ass = np.array(Ass.tolist(), dtype=float)
        Bss = np.array(Bss.tolist(), dtype=float)
        Bdss = np.array(Bdss.tolist(), dtype=float)


        return Ass, Bss, Bdss, Css, Czss, Dss, Dzss



