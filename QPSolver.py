import numpy as np
from cvxopt import solvers, matrix


def qpsolver(H,g,l=None,u=None,A=None,bl=None,bu=None,xinit=None):
        """
        QP solver interface that solves problem
            min x'Hx + g'x
        st.
            l <= x <= u
            bl <= Ax <= bu
        
        Params:
            H:
            g:
            l:
            u:
            A:
            bl:
            bu:
            xinit:
        Outputs:
            x:
            info:
        """
        n = g.shape[0]
        H = matrix(H)
        g = np.array(g).astype(float)
        g = matrix(g)
        if l is not None:
            if A is None:
                G = np.block([[-np.eye(n)],
                            [np.eye(n)]])
                
                h = np.concatenate((-l, u))
            else:
                G = np.block([[-np.eye(n)],
                            [np.eye(n)],
                            [-A],
                            [A],])
                
                h = np.concatenate((-l, u, -bl, bu))
            G = matrix(G)
            h = np.array(h, dtype=np.float64)
            h = matrix(h, tc='d')
            
            sol = solvers.qp(H, g, G, h)
        else:
            sol = solvers.qp(H, g)
        return np.array(sol["x"]).ravel(), sol


