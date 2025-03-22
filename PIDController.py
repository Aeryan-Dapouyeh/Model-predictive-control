import numpy as np

class PIDController():
    def __init__(self, Kp, KI, Kd, I_n, M_unmix, un_Max, e_prev = 0, dt = 1):
        self.I_n_prev = I_n
        self.M_unmix = M_unmix
        self.e_prev = e_prev
        self.dt = dt
        self.Kp = Kp
        self.KI = KI
        self.Kd = Kd
        self.un_Max = un_Max


    def forward(self, zbar, z):
        e_n = zbar - z
        # e_n = self.M_unmix@e_n
        P_n = self.Kp*e_n
        I_n = self.I_n_prev + (self.KI*e_n*self.dt)
        D_n = -self.Kd*(e_n - self.e_prev)/self.dt
        u_n = P_n + I_n + D_n
        self.e_prev = e_n

        if u_n[0] > self.un_Max[0]:
            u_n[0]=self.un_Max[0]

        if u_n[0] > self.un_Max[1]:
            u_n[1]=self.un_Max[1]

        self.I_n_prev = I_n

        # return np.array([u_n[1], u_n[0]])
        return u_n


