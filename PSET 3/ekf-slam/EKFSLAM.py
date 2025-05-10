import numpy as np
import matplotlib.pyplot as plt
from Renderer import Renderer
from Visualization import Visualization

class EKFSLAM(object):
    """A class for implementing EKF-based SLAM

        Attributes
        ----------
        mu :           The mean vector (numpy.array)
        Sigma :        The covariance matrix (numpy.array)
        R :            The process model covariance matrix (numpy.array)
        Q :            The measurement model covariance matrix (numpy.array)
        XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
        MGT :          Ground-truth map (optional, may be None)

        Methods
        -------
        prediction :   Perform the prediction step
        update :       Perform the measurement update step
        augmentState : Add a new landmark(s) to the state
        run :          Main EKF-SLAM loop
        render :       Render the filter
    """

    def __init__(self, mu, Sigma, R, Q, XGT = None, MGT = None):
        """Initialize the class

            Args
            ----------
            mu :           The initial mean vector (numpy.array)
            Sigma :        The initial covariance matrix (numpy.array)
            R :            The process model covariance matrix (numpy.array)
            Q :            The measurement model covariance matrix (numpy.array)
            XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
            MGT :          Ground-truth map (optional, may be None)
        """
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

        self.XGT = XGT
        self.MGT = MGT

        if (self.XGT is not None and self.MGT is not None):
            xmin = min(np.amin(XGT[1, :]) - 2, np.amin(MGT[1, :]) - 2)
            xmax = min(np.amax(XGT[1, :]) + 2, np.amax(MGT[1, :]) + 2)
            ymin = min(np.amin(XGT[2, :]) - 2, np.amin(MGT[2, :]) - 2)
            ymax = min(np.amax(XGT[2, :]) + 2, np.amax(MGT[2, :]) + 2)
            xLim = np.array((xmin, xmax))
            yLim = np.array((ymin, ymax))
        else:
            xLim = np.array((-8.0, 8.0))
            yLim = np.array((-8.0, 8.0))

        self.renderer = Renderer(xLim, yLim, 3, 'red', 'green')

        # Draws the ground-truth map
        if self.MGT is not None:
            self.renderer.drawMap(self.MGT)

        self.MU = mu
        self.VAR = np.diag(self.Sigma).reshape(3, 1)

        # You may find it useful to keep a dictionary that maps a feature ID
        # to the corresponding index in the mean vector and covariance matrix
        self.mapLUT = {}

    def prediction(self, u):
        """Perform the prediction step to determine the mean and covariance
           of the posterior belief given the current estimate for the mean
           and covariance, the control data, and the process model

            Args
            ----------
            u :  The forward distance and change in heading (numpy.array)
        """
        # u is [t, d, dtheta]
        prev_theta = self.angleWrap(self.mu[2])
        self.mu[0] = self.mu[0] + (u[1]) * np.cos(prev_theta)
        self.mu[1] = self.mu[1] + (u[1]) * np.sin(prev_theta)
        self.mu[2] = self.angleWrap(prev_theta + u[2])

        jacobian_F = np.array([
            [1, 0, -u[1] * np.sin(prev_theta)],
            [0, 1, u[1] * np.cos(prev_theta)],
            [0, 0, 1],
        ])
        jacobian_G = np.array([
            [np.cos(prev_theta), 0],
            [np.sin(prev_theta), 0],
            [0, 1],
        ])
        # or just
        # self.Sigma = jacobian_F @ self.Sigma @ jacobian_F.T + self.R

        if len(self.mu) > 3:
            sigma_x_x = self.Sigma[:3, :3]
            sigma_m_x = self.Sigma[3:, :3]
            sigma_x_m = self.Sigma[:3, 3:]
            sigma_m_m = self.Sigma[3:, 3:]

            top_left = jacobian_F @ sigma_x_x @ jacobian_F.T + jacobian_G @ self.R @ jacobian_G.T
            top_right = jacobian_F @ sigma_x_m
            bot_left = (jacobian_F @ sigma_x_m).T
            bot_right = sigma_m_m
            # self.Sigma = np.block([[top_left, top_right],
            #                     bot_left, bot_right])
            upper = np.hstack((top_left, top_right))
            lower = np.hstack((bot_left, bot_right))
            self.Sigma = np.vstack((upper, lower))
        else:
            self.Sigma = jacobian_F @ self.Sigma @ jacobian_F.T + jacobian_G @ self.R @ jacobian_G.T

    def update(self, z, id, t, DEBUG_STEP):
        """Perform the measurement update step to compute the posterior
           belief given the predictive posterior (mean and covariance) and
           the measurement data

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark (int)
        """
        # TODO: Your code goes here
        xr, yr, th = self.mu[:3]
        lm_mu_index = self.mapLUT[id]
        lm_x, lm_y = self.mu[lm_mu_index: lm_mu_index + 2]

        lm_dx = lm_x - xr
        lm_dy = lm_y - yr
        ct, st = np.cos(th), np.sin(th)

        z_hat = np.array([
            ct * lm_dx + st * lm_dy,
            -st * lm_dx + ct * lm_dy,
        ])

        H = np.zeros((2, len(self.mu)))
        H[:, :3] = np.array([
            [-ct, -st, -st*lm_dx + ct*lm_dy],
            [st, -ct, -ct*lm_dx - st*lm_dy],
        ])
        H[:, lm_mu_index:lm_mu_index + 2] = np.array([
            [ct, st],
            [-st, ct],
        ])

        z_diff = z - z_hat
        S = (H @ self.Sigma @ H.T + self.Q)
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        if t >= DEBUG_STEP:
            print("\nInnovation analysis")
            print("ẑ      :", z_hat)
            print("ν=z−ẑ  :", z_diff)
            print("‖ν‖₂    :", np.linalg.norm(z_diff))
            print("S diag  :", np.diag(S))
            print("eig(S)  :", np.linalg.eigvalsh(S))
            print("K max   :", np.abs(K).max())

        self.mu = self.mu + K @ z_diff
        self.mu[2] = self.angleWrap(self.mu[2])

        self.Sigma = (np.eye(len(self.mu)) - K @ H) @ self.Sigma

        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

    def augmentState(self, z, id):
        """Augment the state vector to include the new landmark

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark
        """

        # TODO: Your code goes here
        self.mapLUT[id] = len(self.mu)

        lm_x = self.mu[0] + z[0] * np.cos(self.mu[2]) - z[1] * np.sin(self.mu[2])
        lm_y = self.mu[1] + z[0] * np.sin(self.mu[2] ) + z[1] * np.cos(self.mu[2])

        self.mu = np.append(self.mu, [lm_x, lm_y])

        G_r = np.array([
            [1, 0, - np.sin(self.mu[2]) * z[0] - np.cos(self.mu[2]) * z[1]],
            [0, 1, np.cos(self.mu[2]) * z[0] - np.sin(self.mu[2]) * z[1]]
        ])
        G_z = np.array([
            [np.cos(self.mu[2]), -np.sin(self.mu[2])],
            [np.sin(self.mu[2]), np.cos(self.mu[2])],
        ])
        sigma_x_x = self.Sigma[:3, :3]
        sigma_x_m = self.Sigma[:3, 3:]
        sigma_m_x = self.Sigma[3:, :3]
        sigma_m_m = self.Sigma[3:, 3:]

        sigma_new_new = G_r @ sigma_x_x @ G_r.T + G_z @ self.Q @ G_z.T
        sigma_new_xx = G_r @ sigma_x_x
        sigma_new_xm = G_r @ sigma_x_m
        sigma_xx_new = sigma_x_x @ G_r.T
        sigma_mx_new = sigma_m_x @ G_r.T

        self.Sigma = np.block([
            [sigma_x_x, sigma_x_m, sigma_xx_new],
            [sigma_m_x, sigma_m_m, sigma_mx_new],
            [sigma_new_xx, sigma_new_xm, sigma_new_new],
        ])


    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2*np.pi

        while theta > np.pi:
            theta = theta - 2*np.pi

        return theta

    def _debug_update_preflight(self, lm_id, z):
        xr, yr, th = self.mu[:3]
        idx = self.mapLUT.get(lm_id, None)
        if idx is not None:
            lm = self.mu[idx:idx+2]
        else:
            lm = "(new landmark)"

        print(f"μ prior: {self.mu[:3]}")             # pose *before* update
        print(f"σ diag : {np.diag(self.Sigma)[:6]}") # first few variances
        print(f"id     : {lm_id}")
        print(f"z_meas : {z}")
        print(f"lm μ   : {lm}")

    def run(self, U, Z):
        """The main loop of EKF-based SLAM

            Args
            ----------
            U :   Array of control inputs, one column per time step (numpy.array)
            Z :   Array of landmark observations in which each column
                  [t; id; x; y] denotes a separate measurement and is
                  represented by the time step (t), feature id (id),
                  and the observed (x, y) position relative to the robot
        """
        # TODO: Your code goes here
        DEBUG_STEP = 790

        z_index = 0
        for t, u in enumerate(U.T, start=1):
            print(f"---{t}----{u}----------")
            self.prediction(u)

            z_ind_start = z_index
            while z_index < len(Z[0]) and Z[0, z_index] == t:
                lm_id = str(int(Z[1, z_index]))
                z_meas = Z[2:4, z_index]

                # if t >= DEBUG_STEP:
                #     print("\n*** entering debug step", t, "***")
                #     self._debug_update_preflight(lm_id, z_meas)

                if lm_id not in self.mapLUT:
                    self.augmentState(Z[2:4, z_index], lm_id)
                # else:
                #     self.update(Z[2:4, z_index], lm_id, t, DEBUG_STEP)

                z_index += 1

            if self.MGT is not None:
                self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Z[:, z_ind_start:z_index], self.mapLUT)

            self.MU = np.column_stack((self.MU, self.mu[:3]))
            self.VAR = np.column_stack((self.VAR, np.diag(self.Sigma[:3, :3])))
            print(f"    mu: {self.mu[:3]},  var: {np.diag(self.Sigma[:3, :3])}")

            # self.renderer.render(self.mu, self.Sigma, self.XYT[:, t])

        self.renderer.drawTrajectory(self.MU[0:2, :], self.XGT[1:3, :])
        self.renderer.plotError(self.MU, self.XGT[1:4, :], self.VAR)
        plt.ioff()
        plt.show()
        # You may want to call the visualization function between filter steps where
        #       self.XGT[1:4, t] is the column of XGT containing the pose the current iteration
        #       Zt are the columns in Z for the current iteration
        #       self.mapLUT is a dictionary where the landmark IDs are the keys
        #                   and the index in mu is the value
        #
        # self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Zt, self.mapLUT)
