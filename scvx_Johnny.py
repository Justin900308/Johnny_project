from scipy import signal
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline


# Input

P_des = np.array(([[15, 10]]))
P_ini = np.array(([[0, 0]]))

Ts = 0.1  # Sampling time
T_end = 12  # End time
T = int(T_end / Ts + 1)  # Total time steps

# Specify the obstacles
obs_center = np.zeros((T, 2))
for i in range(T):  # obs centers
    obs_center[i, :] = np.array([5 - 0.0 * i, 5 - 0.00 * i])

R = 3.5  # obs r


def trajectory_gen(P_des, P_ini, obs_center, R, T):
    # Define system matrices
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    C = np.eye(4)
    D = np.zeros((4, 2))

    # Continuous-time system
    sys = signal.StateSpace(A, B, C, D)

    # Discretize the system
    sysd = sys.to_discrete(Ts)
    Ad = sysd.A
    Bd = sysd.B

    # Initialize state and input arrays
    count = 0
    X = np.zeros((4, 1))  # State trajectory
    X[:, 0] = np.array([P_ini[0, 0], P_ini[0, 1], 0, 0])  # Initial state

    u = np.empty((2, 0))  # Initialize u as an empty 2x0 array to store control inputs

    # Compute the initial trajectory
    for t in range(T):
        P_err = P_des - X[0:2, t].T
        u_des = 0.05 * P_err
        u = np.concatenate((u, u_des.T), axis=1)  # Append control input [1, 1]
        # Compute the next state
        X = np.hstack((X, Ad @ X[:, count].reshape(-1, 1) + Bd @ u[:, count].reshape(-1, 1)))
        count += 1

    # Plot the trajectory
    plt.plot(X[0, :], X[1, :], '.')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Trajectory Plot')
    plt.grid(True)
    # plt.show()

    #######################################################################

    alpha = 1
    r_default = 1

    lambda_param = 10000

    # Time length N and trajectory state X
    N = X.shape[1]

    # Convert u to a numpy array for proper matrix operations
    u = np.array(u)
    u = np.reshape(u, (2, int(T_end / Ts + 1)))

    # Plot the obstacle (circle)
    theta = np.linspace(0, 2 * np.pi, 201)
    x_theta = R * np.cos(theta)
    y_theta = R * np.sin(theta)


    for i in range(int(T)):
        plt.plot(X[0, :], X[1, :], '.')
        plt.plot(obs_center[i, 0] + x_theta, obs_center[i, 1] + y_theta)

    # Start the iterative optimization process
    for k in range(30):

        # Define variables for optimization
        w = cp.Variable((2, N - 1))
        v = cp.Variable((4, N - 1))
        d = cp.Variable((4, N))

        s = cp.Variable((1, N - 1))

        # Define the cost function
        Linear_cost = 1 * cp.norm(((u + w) * Ts), 1) + 1 * lambda_param * cp.sum(
            cp.sum(cp.abs(v))) + 1 * lambda_param * cp.sum(cp.pos(s))

        # Define constraints
        constraints = [d[:, 0] == np.zeros(4)]

        E = np.eye(4)

        for i in range(N - 1):
            constraints.append(
                X[:, i + 1] + d[:, i + 1] == (Ad @ X[:, i] + Ad @ d[:, i]) + (Bd @ u[:, i] + Bd @ w[:, i]) + E @ v[:,
                                                                                                                 i])

            # constraints.append(cp.abs(w[0, i]) <= r_default)
            constraints.append(w[0, i] <= r_default)
            constraints.append(-r_default <= w[0, i])
            constraints.append(w[1, i] <= r_default)
            constraints.append(-r_default <= w[1, i])

            # Obstacle avoidance constraint
            constraints.append(2 * R - cp.norm(X[0:2, i] - obs_center[i, :], 2) - (X[0:2, i] - obs_center[i, :]).T @ (
                    X[0:2, i] + d[0:2, i] - obs_center[i, :]) / cp.norm(X[0:2, i] - obs_center[i, :], 2) <= s[:, i])
            constraints.append(s[:, i] >= 0)

        # Terminal condition
        constraints.append(X[:, N - 1] + d[:, N - 1] == np.array([P_des[0, 0], P_des[0, 1], 0, 0]))

        # Define the problem
        problem = cp.Problem(cp.Minimize(Linear_cost), constraints)

        # Solve the optimization problem
        problem.solve(solver=cp.CLARABEL)

        # Update the variables after solving
        w_val = w.value
        v_val = v.value
        d_val = d.value
        s_val = s.value
        # U_val = U.value
        j = 1
        linear_cost = np.zeros((201, 1))
        linear_cost[k, 0] = (1 * LA.norm(((u + w_val) * Ts), ord=1) +
                             1 * lambda_param * np.sum(np.sum(np.abs(v_val))) +
                             1 * lambda_param * np.sum(s_val))

        rho0 = 0
        rho1 = 0.25
        rho2 = 0.7
        delta_L = (linear_cost[k + 1, 0] - linear_cost[k, 0]) / linear_cost[k, 0]
        if delta_L <= rho0:
            r_default = np.max((r_default, 0.8))
            X = X + d_val
            u = u + w_val
        elif delta_L <= rho1:
            r_default = r_default
            X = X + d_val
            u = u + w_val
        elif delta_L <= rho2:
            r_default = r_default / 3.2
            X = X + d_val
            u = u + w_val
        else:
            X = X
            u = u
        r_default = 1

        # Update the trajectory

        # Plot the updated trajectory
        plt.plot(X[0, :], X[1, :], '.')
        plt.pause(0.001)
        # plt.clf()
        ss = np.zeros((T))

        ss_max = np.array([0])
        for i in range(T):
            ss[i] = LA.norm(X[0:2, i] - obs_center[i, :], 2) - R

        if (np.min(ss) > 0) and (k > 4):
            break
        print(np.min(ss) )
        print('Iteration:  ', k + 1)

    # Final trajectory plot
    plt.clf()
    return (X, u)


######################### Main
X, u = trajectory_gen(P_des, P_ini, obs_center, R, T)

T_series = np.zeros((1, T+1))
for i in range(T+1):
    T_series[0, i] = Ts * i

Discre_x=X[0:1,:]
Discre_y=X[1:2,:]



spline_x = InterpolatedUnivariateSpline(T_series, X[0:1,:], k=2)
spline_y = InterpolatedUnivariateSpline(T_series, X[1:2,:], k=2)
spline_u = InterpolatedUnivariateSpline(T_series, X[2:3,:], k=2)
spline_v = InterpolatedUnivariateSpline(T_series, X[3:4,:], k=2)

t_continuous = np.linspace(T_series[0], T_series[-1], 1000)

X_continuous = spline_x(t_continuous)
Y_continuous = spline_y(t_continuous)
u_continuous = spline_u(t_continuous)
v_continuous = spline_v(t_continuous)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot each subplot
axs[0, 0].plot(t_continuous[0,:], X_continuous[0,:], 'r')
axs[0, 0].set_title('X Position data')

axs[0, 1].plot(t_continuous[0,:], Y_continuous[0,:], 'r')
axs[0, 1].set_title('Y Position data')

axs[1, 0].plot(t_continuous[0,:], u_continuous[0,:], 'r')
axs[1, 0].set_title('u data')

axs[1, 1].plot(t_continuous[0,:], v_continuous[0,:], 'r')
axs[1, 1].set_title('v data')

# Add some space between plots and display
plt.tight_layout()
plt.show()








for i in range(int(T)):
    plt.plot(X[0, :i], X[1, :i], '.')
    theta = np.linspace(0, 2 * np.pi, 201)
    x_theta = R * np.cos(theta)
    y_theta = R * np.sin(theta)
    plt.plot(obs_center[i, 0] + x_theta, obs_center[i, 1] + y_theta)
    plt.xlim((P_ini[0, 0] - 0, P_des[0, 0] + 5))
    plt.ylim((P_ini[0, 1] - 0, P_des[0, 1] + 5))
    plt.plot(P_des[0, 0], P_des[0, 1], 'r.', markersize=10)
    plt.pause(0.001)
    plt.clf()
