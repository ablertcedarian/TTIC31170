import numpy as np
from Visualization import Visualization
# import cupy as cp

#matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PF(object):
    """A class for implementing particle filters

        Attributes
        ----------
        numParticles : The number of particles to use
        particles :    A 3 x numParticles array, where each column represents a
                       particular particle, i.e., particles[:,i] = [x^(i), y^(i), theta^(i)]
        weights :      An array of length numParticles array, where each entry
                       denotes the weight of that particular particle
        Alpha :        Vector of 6 noise coefficients for the motion model
                       (See Table 5.3 in Probabilistic Robotics)
        laser :        Instance of the laser class that defines LIDAR params,
                       observation likelihood, and utils
        gridmap :      An instance of the Gridmap class that specifies
                       an occupancy grid representation of the map
                       where 1: occupied and 0: free
        visualize:     Boolean variable indicating whether to visualize
                       the particle filter


        Methods
        -------
        sampleParticlesUniform : Samples a set of particles according to a
                                 uniform distribution
        sampleParticlesGaussian: Samples a set of particles according to a
                                 Gaussian distribution over (x,y) and a
                                 uniform distribution over theta
        getParticle :            Returns the (x, y, theta) and weight associated
                                 with a particular particle id.
        getNormalizedWeights :   Returns the normalized particle weights (numpy.array)
        getMean :                Queries the sample-based estimate of the mean
        prediction :             Performs the prediction step
        update :                 Performs the update step
        run :                    The main loop of the particle filter

    """

    def __init__(self, numParticles, Alpha, laser, gridmap, visualize=True):
        """Initialize the class

            Args
            ----------
            numParticles : The number of particles to use
            Alpha :        Vector of 6 noise coefficients for the motion model
                           (See Table 5.3 in Probabilistic Robotics)
            laser :        Instance of the laser class that defines LIDAR params,
                           observation likelihood, and utils
            gridmap :      An instance of the Gridmap class that specifies
                           an occupancy grid representation of the map
                           here 1: occupied and 0: free
            visualize:     Boolean variable indicating whether to visualize
                           the particle filter (optional, default: True)
        """
        self.numParticles = numParticles
        self.Alpha = Alpha
        self.laser = laser
        self.gridmap = gridmap
        self.visualize = visualize

        # particles is a numParticles x 3 array, where each column denote a particle_handle
        # weights is a numParticles x 1 array of particle weights
        self.particles = None
        self.weights = None

        if self.visualize:
            self.vis = Visualization()
            self.vis.drawGridmap(self.gridmap)
        else:
            self.vis = None

        # Pre-compute batches for collision check to avoid re-allocation in loops
        self.batch_size = min(100, numParticles)  # Adjust based on memory constraints

    def sampleParticlesUniform(self):
        """
            Samples the set of particles according to a uniform distribution and
            sets the weights to 1/numParticles. Particles in collision are rejected
        """

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            theta = np.random.uniform(-np.pi, np.pi)
            inCollision = True
            while inCollision:
                x = np.random.uniform(0, (n-1)*self.gridmap.xres)
                y = np.random.uniform(0, (m-1)*self.gridmap.yres)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))

    def sampleParticlesGaussian(self, x0, y0, sigma):
        """
            Samples the set of particles according to a Gaussian distribution
            Orientation are sampled from a uniform distribution

            Args
            ----------
            x0 :           Mean x-position
            y0  :          Mean y-position
                           (See Table 5.3 in Probabilistic Robotics)
            sigma :        Standard deviation of the Gaussian
        """

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            inCollision = True
            while inCollision:
                x = np.random.normal(x0, sigma)
                y = np.random.normal(y0, sigma)
                theta = np.random.uniform(-np.pi, np.pi)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))

    def getParticle(self, k):
        """
            Returns desired particle (3 x 1 array) and weight

            Args
            ----------
            k :   Index of desired particle

            Returns
            -------
            particle :  The particle having index k
            weight :    The weight of the particle
        """

        if k < self.particles.shape[1]:
            return self.particles[:, k], self.weights[:, k]
        else:
            print('getParticle: Request for k=%d exceeds number of particles (%d)' % (k, self.particles.shape[1]))
            return None, None

    def getNormalizedWeights(self):
        """
            Returns an array of normalized weights

            Returns
            -------
            weights :  An array of normalized weights (numpy.array)
        """

        return self.weights/np.sum(self.weights)

    def getMean(self):
        """
            Returns the mean of the particle filter distribution

            Returns
            -------
            mean :  The mean of the particle filter distribution (numpy.array)
        """

        weights = self.getNormalizedWeights()
        return np.sum(np.tile(weights, (self.particles.shape[0], 1)) * self.particles, axis=1)

    def render(self, ranges, deltat, XGT):
        """
            Visualize filtering strategies

            Args
            ----------
            ranges :   LIDAR ranges (numpy.array)
            deltat :   Step size
            XGT :      Ground-truth pose (numpy.array)
        """

        self.vis.drawParticles(self.particles)
        if XGT is not None:
            self.vis.drawLidar(ranges, self.laser.Angles, XGT[0], XGT[1], XGT[2])
            self.vis.drawGroundTruthPose(XGT[0], XGT[1], XGT[2])
        mean = self.getMean()
        self.vis.drawMeanPose(mean[0], mean[1])
        plt.pause(deltat)

    # def angleWrap(self, theta):
    #     """Ensure that a given angle is in the interval (-pi, pi)."""
    #     while theta < -np.pi:
    #         theta = theta + 2*np.pi

    #     while theta > np.pi:
    #         theta = theta - 2*np.pi

    #     return theta

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        return np.mod(theta + np.pi, 2*np.pi) - np.pi

    def prediction2(self, u, deltat):
        """
            Implement the proposal step using the motion model based in inputs
            v (forward velocity) and w (angular velocity) for deltat seconds

            This is an optimized vectorized implementation

            Args
            ----------
            u :       Two-vector of control inputs (numpy.array)
            deltat :  Step size
        """
        # Generate noise terms for all particles at once
        v_t_1 = np.random.normal(0, self.Alpha[0] * (u[0]**2) + self.Alpha[1] * (u[1]**2), size=self.numParticles)
        v_t_2 = np.random.normal(0, self.Alpha[2] * (u[0]**2) + self.Alpha[3] * (u[1]**2), size=self.numParticles)
        gamma_t = np.random.normal(0, self.Alpha[4] * (u[0]**2) + self.Alpha[5] * (u[1]**2), size=self.numParticles)

        # Process in smaller batches to avoid overwhelming memory
        for batch_start in range(0, self.numParticles, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.numParticles)
            batch_size = batch_end - batch_start

            # Extract current batch
            x_t_1 = self.particles[0, batch_start:batch_end]
            y_t_1 = self.particles[1, batch_start:batch_end]
            theta_t_1 = self.particles[2, batch_start:batch_end]

            # Get noise for this batch
            u_t_1 = u[0] + v_t_1[batch_start:batch_end]
            u_t_2 = u[1] + v_t_2[batch_start:batch_end]
            gamma = gamma_t[batch_start:batch_end]

            # Handle division by zero for u_t_2
            eps = 1e-10
            ratio = np.zeros_like(u_t_2)
            nonzero_mask = np.abs(u_t_2) > eps
            ratio[nonzero_mask] = u_t_1[nonzero_mask] / u_t_2[nonzero_mask]

            # For zero angular velocity, use straight-line motion
            zero_mask = ~nonzero_mask

            # Compute new positions (vectorized)
            theta_new = self.angleWrap(theta_t_1 + u_t_2 * deltat + gamma * deltat)

            # Calculate for particles with non-zero angular velocity
            x_new = np.zeros_like(x_t_1)
            y_new = np.zeros_like(y_t_1)

            if np.any(nonzero_mask):
                x_new[nonzero_mask] = x_t_1[nonzero_mask] + ratio[nonzero_mask] * (
                    np.sin(theta_t_1[nonzero_mask] + u_t_2[nonzero_mask] * deltat) -
                    np.sin(theta_t_1[nonzero_mask])
                )
                y_new[nonzero_mask] = y_t_1[nonzero_mask] + ratio[nonzero_mask] * (
                    -np.cos(theta_t_1[nonzero_mask] + u_t_2[nonzero_mask] * deltat) +
                    np.cos(theta_t_1[nonzero_mask])
                )

            # Calculate for particles with zero angular velocity (straight line)
            if np.any(zero_mask):
                x_new[zero_mask] = x_t_1[zero_mask] + u_t_1[zero_mask] * np.cos(theta_t_1[zero_mask]) * deltat
                y_new[zero_mask] = y_t_1[zero_mask] + u_t_1[zero_mask] * np.sin(theta_t_1[zero_mask]) * deltat

            # Check collisions (vectorized)
            collisions = np.array([self.gridmap.inCollision(x, y) for x, y in zip(x_new, y_new)])
            valid_mask = ~collisions

            # Update only non-colliding particles
            if np.any(valid_mask):
                self.particles[0, batch_start:batch_end][valid_mask] = x_new[valid_mask]
                self.particles[1, batch_start:batch_end][valid_mask] = y_new[valid_mask]
                self.particles[2, batch_start:batch_end][valid_mask] = theta_new[valid_mask]

    def updateWeightsBatch(self, ranges, batch_start, batch_end):
        """
        Update weights for a batch of particles
        """
        batch_weights = np.zeros(batch_end - batch_start)

        for i in range(batch_start, batch_end):
            particle, _ = self.getParticle(i)
            batch_weights[i - batch_start] = self.laser.scanProbability2(ranges, particle, self.gridmap)[0][0]

        return batch_weights

    def prediction(self, u, deltat):
        """
            Implement the proposal step using the motion model based in inputs
            v (forward velocity) and w (angular velocity) for deltat seconds

            This model corresponds to that in Table 5.3 in Probabilistic Robotics

            Args
            ----------
            u :       Two-vector of control inputs (numpy.array)
            deltat :  Step size
        """

        # TODO: Your code goes here: Implement the algorithm given in Table 5.3
        # Note that the "sample" function in the text assumes zero-mean
        # Gaussian noise. You can use the NumPy random.normal() function
        # Be sure to reject samples that are in collision
        # (see Gridmap.inCollision), and to unwrap orientation so that it
        # it is between -pi and pi.

        # Hint: Repeatedly calling np.random.normal() inside a for loop
        #       can consume a lot of time. You may want to consider drawing
        #       n (e.g., n=10) samples of each noise term at once
        #       (drawing n samples is faster than drawing 1 sample n times)
        #       and if none of the estimated poses are not in collision, assume
        #       that the robot doesn't move from t-1 to t.

        v_t_1 = np.random.normal(0, self.Alpha[0] * (u[0]**2) + self.Alpha[1] * (u[1]**2), size=self.numParticles)
        v_t_2 = np.random.normal(0, self.Alpha[2] * (u[0]**2) + self.Alpha[3] * (u[1]**2), size=self.numParticles)
        gamma_t = np.random.normal(0, self.Alpha[4] * (u[0]**2) + self.Alpha[5] * (u[1]**2), size=self.numParticles)

        for particle_index in range(self.numParticles):
            (x_t_1, y_t_1, theta_t_1), weight = self.getParticle(particle_index)

            u_t_1 = u[0] + v_t_1[particle_index]
            u_t_2 = u[1] + v_t_2[particle_index]

            x_t = x_t_1 + (u_t_1 / u_t_2) * (np.sin(theta_t_1 + u_t_2 * deltat) - np.sin(theta_t_1))
            y_t = y_t_1 + (u_t_1 / u_t_2) * (-np.cos(theta_t_1 + u_t_2 * deltat) + np.cos(theta_t_1))
            theta_t_raw = theta_t_1 + u_t_2 * deltat + gamma_t[particle_index] * deltat
            theta_t = self.angleWrap(theta_t_raw)
            # print(f"we wrapped theta: {theta_t_raw} to {theta_t}")

            # Check collision - rejects collided samples by default
            inCollision = self.gridmap.inCollision(x_t, y_t)
            if not inCollision:
                self.particles[0, particle_index] = x_t
                self.particles[1, particle_index] = y_t
                self.particles[2, particle_index] = theta_t

    def resample2(self):
        """
            Perform resampling with replacement - optimized implementation
        """
        # Get normalized weights
        weights_normalized = self.getNormalizedWeights().flatten()

        # Use numpy's fast choice function with probabilities
        indices = np.random.choice(
            self.numParticles,
            size=self.numParticles,
            p=weights_normalized,
            replace=True
        )

        # Create resampled particle set
        self.particles = self.particles[:, indices]

        # Reset weights
        self.weights = np.ones_like(self.weights) / self.numParticles

    def resample(self):
        """
            Perform resampling with replacement
        """

        # TODO: Your code goes here
        # The np.random.choice function may be useful

        choices = np.random.choice(
            self.numParticles,
            size=self.numParticles,
            p=self.getNormalizedWeights().flatten(),
        )
        self.particles = (self.particles.T[choices]).T

    def update2(self, ranges):
        """
            Implement the measurement update step

            Args
            ----------
            ranges :    Array of LIDAR ranges (numpy.array)
        """
        # Process particles in batches
        all_weights = np.zeros(self.numParticles)

        for batch_start in range(0, self.numParticles, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.numParticles)

            # Update weights for this batch
            batch_weights = self.updateWeightsBatch(ranges, batch_start, batch_end)
            all_weights[batch_start:batch_end] = batch_weights

        # Reshape and assign weights
        self.weights = all_weights.reshape(1, -1)

        # Prevent numerical underflow
        if np.max(self.weights) < 1e-100:
            self.weights = np.ones_like(self.weights) / self.numParticles


    def update(self, ranges):
        """
            Implement the measurement update step

            Args
            ----------
            ranges :    Array of LIDAR ranges (numpy.array)
        """
        # TODO: Your code goes here
        self.weights = np.array([[
            self.laser.scanProbability(ranges, self.getParticle(k)[0], self.gridmap)[0][0]
            for k in range(self.numParticles)
        ]])

    def run2(self, U, Ranges, deltat, X0, XGT, filename):
        """
            The main loop that runs the particle filter

            Args
            ----------
            U :      An array of control inputs, one column per time step (numpy.array)
            Ranges : An array of LIDAR ranges (numpy,array)
            deltat : Duration of each time step
            X0 :     The initial pose (may be None) (numpy.array)
            XGT :    An array of ground-truth poses (may be None) (numpy.array)
        """
        # Initialize particles
        sampleGaussian = True
        NeffThreshold = self.numParticles/10

        if sampleGaussian and (X0 is not None):
            sigma = 0.5
            self.sampleParticlesGaussian(X0[0, 0], X0[1, 0], sigma)
        else:
            self.sampleParticlesUniform()

        # Iterate over the data
        for k in range(U.shape[1]):
            print(f"Time step {k}")
            u = U[:, k]
            ranges = Ranges[:, k]

            # Prediction step
            self.prediction2(u, deltat)

            # Update step
            self.update2(ranges)

            # Calculate effective sample size
            Neff = 1 / np.sum(self.getNormalizedWeights()**2)

            # Resample if necessary
            if Neff < NeffThreshold:
                self.resample2()
                print(f"    Resampling with Neff = {Neff} < {NeffThreshold}")

        # Visualize if needed
        if self.visualize:
            if XGT is None:
                self.render(ranges, deltat, None)
            else:
                self.render(ranges, deltat, XGT[:, k])

        # Save final result
        plt.savefig(filename, bbox_inches='tight')

    def run(self, U, Ranges, deltat, X0, XGT, filename):
        """
            The main loop that runs the particle filter

            Args
            ----------
            U :      An array of control inputs, one column per time step (numpy.array)
            Ranges : An array of LIDAR ranges (numpy,array)
            deltat : Duration of each time step
            X0 :     The initial pose (may be None) (numpy.array)
            XGT :    An array of ground-truth poses (may be None) (numpy.array)
        """

        # TODO: Try different sampling strategies (including different values for sigma)
        sampleGaussian = True
        NeffThreshold = self.numParticles/10
        if sampleGaussian and (X0 is not None):
            sigma = 0.5
            self.sampleParticlesGaussian(X0[0, 0], X0[1, 0], sigma)
        else:
            self.sampleParticlesUniform()

        resample_mode = "baseline"

        # Iterate over the data
        for k in range(U.shape[1]):
            print(f"Time step {k}")
            u = U[:, k]
            ranges = Ranges[:, k]

            # TODO: Your code goes here
            self.prediction(u, deltat)
            self.update(ranges)

            if resample_mode == "baseline":
                Neff = 1 / np.sum(self.getNormalizedWeights()**2)
                if Neff < NeffThreshold:
                    self.resample()
                    print(f"    Resampling with Neff = {Neff} < {NeffThreshold}")
            elif resample_mode == "threshold":
                Neff = 1 / np.sum(self.getNormalizedWeights()**2)
                NeffThreshold = self.numParticles * (2/3)
                if Neff < NeffThreshold:
                    self.resample()
                    print(f"    Resampling by threshold with Neff = {Neff} < {NeffThreshold}")

            if self.visualize:
                if XGT is None:
                    self.render(ranges, deltat, None)
                else:
                    self.render(ranges, deltat, XGT[:, k])

        plt.savefig(filename + f'_{resample_mode}', bbox_inches='tight')
