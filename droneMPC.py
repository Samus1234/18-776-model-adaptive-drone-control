# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:26:05 2021

@author: siddg
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import control
from scipy.linalg import sqrtm
from scipy.linalg import pinv
from qpsolvers import solve_qp

"""
Setting up basic functions for readability
"""

pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
tan = np.tan
inv = np.linalg.inv
norm = np.linalg.norm
atan = np.arctan
sign = np.sign
sqrt = np.sqrt
asin = np.arcsin
acos = np.arccos
mat = np.matmul
matpow = np.linalg.matrix_power
"""
Nonlinear State Estimator
"""

class Estimator(object):
    def __init__(self, x_hat, u, y, P, Q, R, f_sys, h, Ts):
        self.Ts = Ts
        self.Q = Q
        self.R = R
        self.n = np.shape(Q)[0]
        self.k = np.shape(R)[0]
        self.x_hat = x_hat
        self.u = u
        self.P = P
        self.y = y
        self.f_sys = lambda x, u: f_sys(x, u)
        self.f = lambda x: f_sys(x, self.u)
        self.h = lambda x: h(x)
        
    def update(self, y, u):
        self.y = y
        self.u = u
    
    def sigmaPoints(self):
        k = 2
        n = self.n
        X = np.zeros([2*n+1, n])
        W = np.ones(2*n+1)*(0.5/(n+k))
        W[n] *= 2*k
        sqrtP = sqrtm(self.P)
        for i in range(2*n+1):
            X[i] = self.x_hat + sign(n - i) * sqrt(n + k) * sqrtP[i%n]
        return X, W

    def computeJacobian(self, fs, X_op):
        N = len(X_op)
        M = len(fs(X_op))
        J = np.zeros((N, M))
        I = np.identity(N)
        epsilon = 1e-6
        
        for i in range(N):
            J[i, :] = (fs(X_op + epsilon*I[i, :]) - fs(X_op - epsilon*I[i, :]))/2/epsilon
            
        return np.transpose(J)

    
    def meanVarFH(self, X, W):
        n = self.n
        k = self.k
        l = 2*n + 1
        Z = np.zeros([l, n + k])
        for i in range(l):
            Z[i] = np.concatenate((self.f(X[i]), self.h(X[i])))
        meanZ = np.average(Z.T, axis = 1, weights = W)
        covZ = np.cov(Z.T, ddof = 0, aweights = W)
        return meanZ[:n], meanZ[n:], covZ[:n, :n] + self.Q, covZ[n:, n:] + self.R, covZ[:n, n:]
    
    def UKF(self):
       X_apri, W_apri = self.sigmaPoints()
       x_post, y_pred, P_post, S, T = self.meanVarFH(X_apri, W_apri)
       self.x_hat = x_post
       self.P = P_post
       X_post, W_post = self.sigmaPoints()
       _, _, _, S, T = self.meanVarFH(X_post, W_post)
       K = mat(T, inv(S))
       self.x_hat = x_post + mat(K, (self.y - y_pred))
       self.P = P_post - mat(K, T.T)
       
    def EKF(self):
        F = self.computeJacobian(self.f, self.x_hat)
        H = self.computeJacobian(self.h, self.x_hat)
        x_hat_prev = self.f_sys(self.x_hat, self.u)
        P_prev = np.matmul(F, np.matmul(self.P, F.T)) + self.Q
        Kg = np.matmul(P_prev, np.matmul(H.T, inv(self.R + np.matmul(H, np.matmul(P_prev, H.T)))))
        self.x_hat = x_hat_prev + np.matmul(Kg, (self.y - self.h(x_hat_prev)))
        self.P = np.matmul((np.eye(self.n) - np.matmul(Kg, H)), P_prev)
        
        
"""
Here, I must mention that the order of the state-vector is slightly different in the code.
In the report, the state-vector is X = [x y z x_dot y_dot z_dot phi theta psi phi_dot theta_dot psi_dot]
In the code, the state-vector is X = [x x_dot y y_dot z z_dot phi phi_dot theta theta_dot  psi psi_dot]
All the functions will have repective changes in order, but apart from that everything is the same
"""

#Time step size

Ts = 1e-2
freq = 5

# Iniital Condition
X0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Initial estimation state

X0_hat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Kalman Filter Tuning Parameters

P = np.eye(12)*0.1

Q = np.eye(12)*1e-4

R = np.eye(6)*0.5

# Drone Parameters
m = 0.468
g = 9.8
Kf = 2.980*(10**-3)
Km = 2.980*(10**-3)
k = Km/Kf
l = 0.225
b = 0.010
Ixx = 4.856e-3*1.5
Iyy = 4.856e-3*1.5
Izz = 8.801e-3*1.5

# Numerically Compute Jacobian of a given function around an operating point
def computeJacobian(fs, X_op):
    N = len(X_op)
    J = np.zeros((N, N))
    I = np.identity(N)
    epsilon = 1e-6
    
    for i in range(N):
        J[i, :] = (fs(X_op + epsilon*I[i, :]) - fs(X_op - epsilon*I[i, :]))/2/epsilon
        
    return np.transpose(J)

# Compute Lifted Dynamic Matrices for MPC without State constraints
def mpcMatrices(A, B, Q, R, N):
    n, m = np.shape(B)
    l = 8
    limit = 1.25
    Lu = np.array([[1, 0, 0, 0],
                   [-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, -1]])
    bu = np.ones(8)*(1/2/Kf)*limit
    A_aug = np.zeros([n*(N+1), n])
    B_aug = np.zeros([n*(N+1), m*N])
    Q_aug = np.zeros([n*(N+1), n*(N+1)])
    R_aug = np.zeros([m*(N), m*(N)])
    Lu_aug = np.zeros([l*N, m*N])
    Bu_aug = np.zeros([l*N])
    
    A_aug[:n, :n] = np.eye(n)
    for i in range(1, N+1):
        for j in range(i):
            B_aug[n*i:n*(i+1), m*j:m*(j+1)] = mat(matpow(A, (i-(j+1))), B)
        A_aug[n*i:n*(i+1), :n] = matpow(A, i)
        Q_aug[n*(i-1):n*(i), n*(i-1):n*i] = Q
        R_aug[m*(i-1):m*(i), m*(i-1):m*i] = R
        Lu_aug[l*(i-1):l*i, m*(i-1):m*i] = Lu
        Bu_aug[l*(i-1):l*i] = bu
    Q_aug[n*(N):n*(N+1), n*(N):n*(N+1)] = 10*Q
    P_qp = R_aug + mat(B_aug.T, mat(Q_aug, B_aug))
    q_qp = mat(B_aug.T, mat(Q_aug, A_aug))
    G_qp = Lu_aug
    h_qp = Bu_aug
    return A_aug, B_aug, Q_aug, R_aug, Lu_aug, Bu_aug, P_qp, q_qp, G_qp, h_qp

w = 2*pi*freq/100
tau = 10

xyA = 20

zH = 20

trajectory = lambda t: np.array([xyA*cos(w*t - pi/2)/(1 + sin(w*t - pi/2)**2)*(1 - exp(-t/tau)),
                                 xyA*sin(w*t - pi/2)*cos(w*t - pi/2)/(1 + sin(w*t - pi/2)**2)*(1 - exp(-t/tau)),
                                 zH*(1 - exp(-t/tau))])

trajectory_dot = lambda t: np.array([xyA*(-2*w*(-5 + cos(2*w*t))*cos(w*t)/(3 + cos(2*w*t))**2)*(1 - exp(-t/tau)) + xyA*cos(w*t - pi/2)/(1 + sin(w*t - pi/2)**2)*exp(-t/tau)/tau, 
                                     xyA*(-2*w*(1 + 3*cos(2*w*t))/(3 + cos(2*w*t))**2)*(1 - exp(-t/tau)) + xyA*sin(w*t - pi/2)*cos(w*t - pi/2)/(1 + sin(w*t - pi/2)**2)*exp(-t/tau)/tau, 
                                     zH*exp(-t/tau)/tau])

trajectory_ddot = lambda t: np.array([xyA*(-w*w*(2*sin(w*t) - 45*sin(3*w*t) + sin(5*w*t))/(3 + cos(2*w*t))**3/2) + xyA*(-2*w*(-5 + cos(2*w*t))*cos(w*t)/(3 + cos(2*w*t))**2)*exp(-t/tau)/tau - xyA*cos(w*t - pi/2)/(1 + sin(w*t - pi/2)**2)*exp(-t/tau)/tau/tau,
                                      xyA*(4*w*w*(7 - 3*cos(2*w*t))*sin(2*w*t))/(3 + cos(2*w*t))**3 + xyA*(-2*w*(1 + 3*cos(2*w*t))/(3 + cos(2*w*t))**2)*exp(-t/tau)/tau - xyA*sin(w*t - pi/2)*cos(w*t - pi/2)/(1 + sin(w*t - pi/2)**2)*exp(-t/tau)/tau/tau,
                                      -zH*exp(-t/tau)/tau/tau])

# Transform matrix from control inputs to system forces
T_transform = np.array([[Kf, Kf, Kf, Kf], 
                        [0, -l*Kf, 0, l*Kf], 
                        [-l*Kf, 0, l*Kf, 0], 
                        [-Km, Km, -Km, Km]
                        ])
# Inertia Tensor
I = np.array([[Ixx, 0, 0], 
              [0, Iyy, 0], 
              [0, 0, Izz]
              ])
# Drag Coeffiecient
A_drag = np.array([[0.10, 0, 0], 
                   [0, 0.10, 0], 
                   [0, 0, 0.10]
                   ])
# Yaw rotation matrix
R_psi = lambda X: np.array([[cos(X[10]), sin(X[10]), 0], 
                            [-sin(X[10]), cos(X[10]), 0], 
                            [0, 0, 1]
                            ])
# Pitch rotation matrix
R_theta = lambda X: np.array([[cos(X[8]), 0, -sin(X[8])], 
                              [0, 1, 0], 
                              [sin(X[8]), 0, cos(X[8])]
                              ])
# Roll rotation matrix
R_phi = lambda X: np.array([[1, 0, 0], 
                            [0, cos(X[6]), sin(X[6])], 
                            [0, -sin(X[6]), cos(X[6])]
                            ])
# Euler rates to Angular Velocity
W_n = lambda X: np.array([[1, 0, -sin(X[8])], 
                          [0, cos(X[6]), cos(X[8])*sin(X[6])], 
                          [0, -sin(X[6]), cos(X[8])*cos(X[6])]
                          ])
# Inertial to Body Rotation						  						  
R_ib = lambda X: np.matmul(R_phi(X), np.matmul(R_theta(X), R_psi(X)))
# Body to Inertial Rotation
R_bi = lambda X: np.transpose(R_ib(X))
# Exogenous Force on Drone
F_ext = lambda X, w: np.matmul(R_bi(X), np.array([0, 0, np.matmul(T_transform, (w))[0]]))
# Exogenous Moment on Drone
T_ext = lambda X, w: np.matmul(R_bi(X), np.matmul(T_transform, (w))[1:])
# Rotational matrix
J = lambda X: np.matmul(np.transpose(W_n(X)), np.matmul(I, W_n(X)))
# Coriolis matrix
J_dot = lambda X: np.array([
                            [0, 0, -Ixx*cos(X[8])*X[9]], 
                            [0, (Izz - Iyy)*sin(2*X[6])*X[7], (Iyy - Izz)*(-cos(X[6])*sin(X[8])*sin(X[6])*X[9] + cos(X[8])*cos(2*X[6])*X[7])],
                            [-Ixx*cos(X[8])*X[9], (Iyy - Izz)*(-cos(X[6])*sin(X[8])*sin(X[6])*X[9] + cos(X[8])*cos(2*X[6])*X[7]), -2*cos(X[8])*(sin(X[8])*(-Ixx + Izz*cos(X[6])**2 + Iyy*sin(X[6])**2)*X[9] + (Izz - Iyy)*cos(X[8])*cos(X[6])*sin(X[6])*X[7])]
                           ])
# Generalized translational force
Q_trans = lambda X: np.array([0, 
                              0, 
                              -m*g
                              ])
# Generalized rotational force
Q_rot = lambda X: np.array([-(Iyy - Izz)*(sin(2*X[6])*X[9]**2 - X[9]*X[11]*cos(2*X[6])*cos(X[8]) - cos(X[6])*sin(X[6])*cos(X[8])**2*X[11]**2), 
                            X[11]*(-2*Ixx*X[7]*cos(X[8]) + sin(X[8])*((Izz - Iyy)*sin(2*X[6])*X[9] + 2*cos(X[8])*(Ixx - Izz*cos(X[6])**2 - Iyy*sin(X[6])**2)*X[11])),
                            0
                            ])
# xi_ddot
F_x = lambda X, w: (1/m) * (-np.matmul(A_drag, X[1::2][:3]) + Q_trans(X) + F_ext(X, w))
# eta_ddot
F_eta = lambda X, w: np.linalg.solve(J(X), (-np.matmul(J_dot(X), X[1::2][3:6]) + Q_rot(X) + T_ext(X, w)))
# System function
def fs(X):
    X_dot = np.zeros(len(X))
    X_dot[::2] = X[1::2]
    X_dot[1::2][:3] = (1/m) * (-np.matmul(A_drag, X[1::2][:3]) + Q_trans(X))
    X_dot[1::2][3:6] = np.linalg.solve(J(X), (-np.matmul(J_dot(X), X[1::2][3:6]) + Q_rot(X)))
    return X_dot
# Control function
def gs(X):
    G = np.zeros([len(X), 4])
    G[1::2, :][:3, :] = np.matmul(R_bi(X)/m, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [Kf, Kf, Kf, Kf]]))
    G[1::2, :][3:6, :] = np.matmul(np.linalg.solve(J(X), R_bi(X)), np.array([[0, -l*Kf, 0, l*Kf], [-l*Kf, 0, l*Kf, 0], [-Km, Km, -Km, Km]]))
    return G

def f_sys(x, u):
    return x + Ts*(fs(x) + np.matmul(gs(x), u))

A = computeJacobian(lambda x: f_sys(x, (m*g/4/Kf)*np.ones(4)), np.zeros(12))

B = Ts*gs(np.zeros(12))

Q_mpc = 150*np.diag([200, 5, 200, 5, 120, 4, 15, 1, 15, 1, 15, 1])

R_mpc = 1e-5*np.eye(4)

N_horizon = 50

A_aug, B_aug, Q_aug, R_aug, Lu_aug, Bu_aug, P_qp, q_qp, G_qp, h_qp = mpcMatrices(A, B, Q_mpc, R_mpc, N_horizon)

def mpc(t, x):
    x_ref = np.zeros(12)
    x_ref[::2][:3] = trajectory(t)
    return solve_qp(P_qp, mat(q_qp, x - x_ref), G_qp, h_qp) + (m*g/4/Kf)*np.ones(4*N_horizon)


# Wrapper function for ODE solver
def F(t, x):
    u = mpc(t, x)
    return f_sys(x, u[:4])

N = 5000

t = np.arange(N)*Ts

X = np.zeros([N, 12])

X_traj = np.zeros([N, 12*(N_horizon+1)])

X_traj[0] = mat(A_aug, X0)

U = np.zeros([N, 4*N_horizon])

X[0] = X0

U[0] = mpc(0, X0)

for k in range(N-1):
    U[k+1] = mpc(k//N_horizon*Ts, X[k//N_horizon]) + (m*g/4/Kf)*np.ones(4*N_horizon)
    X[k+1] = F(k*Ts, X[k])
    X_traj[k+1] = mat(A_aug, X[k+1])
    
X_pred = X_traj.reshape(N, N_horizon+1, 12)
U_pred = U.reshape(N, N_horizon, 4) 
    
    
"""
Separate Variables for plotting
"""

x = X[:, 0]
y = X[:, 2]
z = X[:, 4]

phi = X[:, 6]
theta = X[:, 8]
psi = X[:, 10]

xdot = X[:, 1]
ydot = X[:, 3]
zdot = X[:, 5]

phidot = X[:, 7]
thetadot = X[:, 9]
psidot = X[:, 11]

xSet = trajectory(t)[0]
ySet = trajectory(t)[1]
zSet = trajectory(t)[2]

"""
Tracking Errors
"""

x_mse = np.sqrt(np.mean(np.square(xSet - x)))
y_mse = np.sqrt(np.mean(np.square(ySet - y)))
z_mse = np.sqrt(np.mean(np.square(zSet - z)))

"""
Plot Output Variables
"""

def plotOutput():
    plt.figure(figsize=(16, 9))
    plt.subplot(211)
    plt.plot(t, x, label = '$x$')
    plt.plot(t, y, label = '$y$')
    plt.plot(t, z, c = '#e62e00', label = '$z$')
    plt.plot(t, xSet, label = '$x_{Traj}$', lw = 2.5, linestyle = '--', c = 'tab:blue')
    plt.plot(t, ySet, label = '$y_{Traj}$', lw = 2.5, linestyle = '--', c = 'tab:orange')
    plt.plot(t, zSet, label = '$z_{Traj}$', linestyle = '--', c = 'tab:red', lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('distance in $m$', fontsize = 18)
    plt.title('Translational coordinates', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.grid()
    
    plt.subplot(212)
    plt.plot(t, 180*phi/pi, label = '$\phi$', lw = 2.5)
    plt.plot(t, 180*theta/pi, label = '$\Theta$', lw = 2.5)
    plt.plot(t, 180*psi/pi, c = '#e62e00', label = '$\psi$', lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('angle in $^\circ$', fontsize = 18)
    plt.title('Rotational coordinates', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.grid()
    
    plt.figure(figsize=(16, 9))
    plt.subplot(211)
    plt.plot(t, xdot, label = '$\dot{x}$', lw = 2.5)
    plt.plot(t, ydot, label = '$\dot{y}$', lw = 2.5)
    plt.plot(t, zdot, c = '#e62e00', label = '$\dot{z}$', lw = 2.5)
    plt.plot(t, trajectory_dot(t)[0], label = '$vx_{Traj}$', lw = 2.5, linestyle = '--', c = 'tab:blue')
    plt.plot(t, trajectory_dot(t)[1], label = '$vy_{Traj}$', lw = 2.5, linestyle = '--', c = 'tab:orange')
    plt.plot(t, trajectory_dot(t)[2], label = '$vz_{Traj}$', linestyle = '--', c = 'tab:red', lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('distance in $m/s$', fontsize = 18)
    plt.title('Translational velocities', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.grid()
    
    plt.subplot(212)
    plt.plot(t, 180*phidot/pi, label = '$\dot{\phi}$', lw = 2.5)
    plt.plot(t, 180*thetadot/pi, label = '$\dot{\Theta}$', lw = 2.5)
    plt.plot(t, 180*psidot/pi, c = '#e62e00', label = '$\dot{\psi}$', lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('angular rate in $^\circ/s$', fontsize = 18)
    plt.title('Rotational velocities', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.grid()

    plt.show()
    
def plotControlEffort():
    plt.figure(figsize=(16, 9))
    plt.subplot(221)
    plt.plot(t, np.sqrt(U[:, 0])/pi/2*60, lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.title('Rotor - 1 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    plt.subplot(222)
    plt.plot(t, np.sqrt(U[:, 1])/pi/2*60, lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.title('Rotor - 2 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    plt.subplot(223)
    plt.plot(t, np.sqrt(U[:, 2])/pi/2*60, lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.title('Rotor - 3 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    plt.subplot(224)
    plt.plot(t, np.sqrt(U[:, 3])/pi/2*60, lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.title('Rotor - 4 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    
    
"""
Plot 3D Trajectory
"""
    
def plot3DTrajectory():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    # Data for a three-dimensional line
    zline = z
    xline = x
    yline = y
    ax.plot3D(xline, yline, zline, 'tab:blue', lw = 2.5, label = 'Actual Drone Trajectory')
    zdata = zSet[::100]
    xdata = xSet[::100]
    ydata = ySet[::100]
    ax.scatter3D(xdata, ydata, zdata, c='tab:red', s = 50, label = 'Waypoints');
    plt.title('Drone 3D Cascade Controller Trajectory Tracking', fontsize = 18)
    ax.set_xlabel('x', fontsize = 18)
    ax.set_ylabel('y', fontsize = 18)
    ax.set_zlabel('z', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()
    
plotOutput()

plot3DTrajectory()

plotControlEffort()


K = 10

T = 80
x_c = x[::K]
y_c = y[::K]
z_c = z[::K]

a = phi[::K]
b = theta[::K]
c = psi[::K]

l = 4.0

x_x1 = x_c + l*cos(b)*cos(c)
y_x1 = y_c + l*cos(b)*sin(c)
z_x1 = z_c + (-l)*sin(b)

x_x2 = x_c - l*cos(b)*cos(c)
y_x2 = y_c - l*cos(b)*sin(c)
z_x2 = z_c - (-l)*sin(b)

x_y1 = x_c + l*cos(c)*sin(b)*sin(a) - l*cos(a)*sin(c)
y_y1 = y_c + l*cos(a)*cos(c) + l*sin(b)*sin(a)*sin(c)
z_y1 = z_c + l*cos(b)*sin(a)

x_y2 = x_c - l*cos(c)*sin(b)*sin(a) + l*cos(a)*sin(c)
y_y2 = y_c - l*cos(a)*cos(c) - l*sin(b)*sin(a)*sin(c)
z_y2 = z_c - l*cos(b)*sin(a)

x_pred = X_pred[::K, :, 0]
y_pred = X_pred[::K, :, 2]
z_pred = X_pred[::K, :, 4]


fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(projection='3d')

i = 0

line1, = ax.plot(np.array([x_x2[i], x_x1[i]]), np.array([y_x2[i], y_x1[i]]), np.array([z_x2[i], z_x1[i]]), c = 'k', lw = 4, marker = 'o')
line2, = ax.plot(np.array([x_y2[i], x_y1[i]]), np.array([y_y2[i], y_y1[i]]), np.array([z_y2[i], z_y1[i]]), c = 'k', lw = 4, marker = 'o')
line3, = ax.plot(x_c[:i], y_c[:i], z_c[:i], label = 'Drone Trajectory', lw = 2.5)
line4, = ax.plot(x_pred[i], y_pred[i], z_pred[i], c = 'r', lw = 2.5, label = 'MPC Predicted Trajectories')
ax.plot3D(xSet, ySet, zSet, c='tab:orange', linestyle = '--', label = 'Waypoints', lw = 0.5);

plt.legend()

def makeFrame(i, line1, line2, line3, line4):
    line1.set_data(np.array([x_x2[i], x_x1[i]]), np.array([y_x2[i], y_x1[i]]))
    line1.set_3d_properties(np.array([z_x2[i], z_x1[i]]))
    line2.set_data(np.array([x_y2[i], x_y1[i]]), np.array([y_y2[i], y_y1[i]]))
    line2.set_3d_properties(np.array([z_y2[i], z_y1[i]]))
    line3.set_data(x_c[:i], y_c[:i])
    line3.set_3d_properties(z_c[:i])
    line4.set_data(x_pred[i], y_pred[i])
    line4.set_3d_properties(z_pred[i])
    
# Setting the axes properties
ax.set_xlim3d([-xyA-5, xyA+5])
ax.set_xlabel('X')

ax.set_ylim3d([-xyA-5, xyA+5])
ax.set_ylabel('Y')

ax.set_zlim3d([-5, zH+5])
ax.set_zlabel('Z')
    
ani = animation.FuncAnimation(fig, makeFrame, N//K, fargs=(line1, line2, line3, line4), interval=1000/120, blit=False)

ani.save('droneMPCWithControlConstraintsOnly.gif', writer='imagemagick')