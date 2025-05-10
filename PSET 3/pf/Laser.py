import numpy as np
from scipy.stats import norm, expon
import numba as nb

# Optional: If available, use numba to compile the ray tracing function
try:
    from numba import jit
    use_numba = True
except ImportError:
    print("Numba not available - falling back to non-compiled version")
    use_numba = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class Laser(object):
    """A class specifying a LIDAR beam model based on Section 6.3.1 of Probabilistic Robotics,
       which is comprised of a mixture of different components, whose parameters are described below.

       Note that we use pX to denote the weight associated with the X component,
       whereas the text uses zX to denote the weight

        Attributes
        ----------
        pHit :        Likelihood (weight) of getting a valid return (subject to noise)
        pShort :      Likelihood (weight) of getting a short return
        pMax :        Likelihood (weight) of getting a false max range return
        pRand :       Likelihood (weight) of a random range in interval [0, zMax]
        sigmaHit :    Standard deviation of the Gaussian noise that corrupts true range
        lambdaShort : Parameter of model determining likelihood of a short return
                      (e.g., due to an object not in the map)
        zMax:         Maximum sensor range
        zMaxEps :


        Methods
        -------
        scanProbability : Computes the likelihood of a given LIDAR scan from
                          a given pose in a given map
        getXY :           Function that converts the range and bearing to Cartesian
                          coordinates in the LIDAR frame
        rayTracing :      Perform ray tracing from a given pose to predict range and bearing returns
    """

    def __init__(self, numBeams=41, sparsity=1):
        """Initialize the class

            Args
            ----------
            numBeams :    Number of beams in an individual scan (optional, default: 41)
            sparsity :    Downsample beams by taking every sparsity-1 beam (optional, default: 1)
        """
        self.pHit = 0.97
        self.pShort = 0.01
        self.pMax = 0.01
        self.pRand = 0.01
        self.sigmaHit = 0.5
        self.lambdaShort = 1
        self.zMax = 20
        self.zMaxEps = .1
        self.Angles = np.linspace(-np.pi, np.pi, numBeams)# array of angles
        self.Angles = self.Angles[::sparsity]

        # Pre-compute for efficiency
        self.normal = norm(0, self.sigmaHit)
        self.exponential = expon(scale=1/self.lambdaShort)

        # Cache for ray tracing results - significant speedup for static maps
        self.ray_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Maximum cache size to prevent memory issues
        self.max_cache_size = 10000

    def scanProbability(self, z, x, gridmap):
        """The following computes the likelihood of a given LIDAR scan from
           a given pose in the provided map according to the algorithm given
           in Table 6.1 of Probabilistic Robotics

            Args
            -------
            z :           An array of LIDAR ranges, one entry per beam (numpy.array)
            x :           An array specifying the LIDAR (x, y, theta) pose (numpy.array)
            gridmap :     The map of the environment as a gridmap

            Returns
            -------
            likelihood :  The probability of the scan.
        """

        # TODO: Your code goes here
        # Implement the algorithm given in Table 6.1
        # You are provided with an implementation (albeit slow) of ray tracing below
        q = 1
        for k in range(len(z)):
            z_k_star_range, z_k_star_coords = self.rayTracing(x[0], x[1], x[2], self.Angles[k], gridmap)
            p = (
                self.pHit * (self.normal.pdf(z[k] - z_k_star_range) if (0 <= z[k] and z[k] <= self.zMax) else 0) +
                self.pShort * (self.exponential.pdf(z_k_star_range) if (0 <= z[k] and z[k] <= z_k_star_range) else 0) +
                self.pMax * (1 if z[k] == self.zMax else 0) +
                self.pRand * (1 / self.zMax if (0 <= z[k] and z[k] <= self.zMax) else 0)
            )
            q *= p
        return q

    def scanProbability2(self, z, x, gridmap):
        """The following computes the likelihood of a given LIDAR scan from
           a given pose in the provided map according to the algorithm given
           in Table 6.1 of Probabilistic Robotics.

           This is an optimized version that vectorizes calculations when possible.

            Args
            -------
            z :           An array of LIDAR ranges, one entry per beam (numpy.array)
            x :           An array specifying the LIDAR (x, y, theta) pose (numpy.array)
            gridmap :     The map of the environment as a gridmap

            Returns
            -------
            likelihood :  The probability of the scan.
        """
        # Initialize probability
        q = np.ones((1, 1))

        # Get expected ranges from ray tracing
        z_star_all = self.batchRayTracing(x[0], x[1], x[2], self.Angles, gridmap)

        # Calculate probabilities for each beam
        for k in range(len(z)):
            z_k = z[k]
            z_k_star = z_star_all[k]

            # pHit component (measurement matches expectation with Gaussian noise)
            p_hit = 0
            if 0 <= z_k <= self.zMax:
                p_hit = self.normal.pdf(z_k - z_k_star)

            # pShort component (short return)
            p_short = 0
            if 0 <= z_k <= z_k_star:
                p_short = self.exponential.pdf(z_k)

            # pMax component (max range return)
            p_max = 1 if z_k == self.zMax else 0

            # pRand component (random noise)
            p_rand = 0
            if 0 <= z_k <= self.zMax:
                p_rand = 1 / self.zMax

            # Combine all probabilities using the mixture weights
            p = (
                self.pHit * p_hit +
                self.pShort * p_short +
                self.pMax * p_max +
                self.pRand * p_rand
            )

            # Update total probability
            q *= p

        return q

    def getXY(self, range, bearing):
        """Function that converts the range and bearing to
           Cartesian coordinates in the LIDAR frame

            Args
            -------
            range :   A 1 x n array of LIDAR ranges (numpy.ndarray)
            bearing : A 1 x n array of LIDAR bearings (numpy.ndarray)

            Returns
            -------
            XY :      A 2 x n array, where each column is an (x, y) pair
        """

        CosSin = np.vstack((np.cos(bearing[:]), np.sin(bearing[:])))
        XY = np.tile(range, (2, 1))*CosSin

        return XY

    def rayTracing(self, xr, yr, thetar, lAngle, gridmap):
        """A vectorized implementation of ray tracing

            Args
            -------
            (xr, yr, thetar):   The robot's pose
            lAngle:             The LIDAR angle (in the LIDAR reference frame)
            gridmap:            An instance of the Gridmap class that specifies
                                an occupancy grid representation of the map
                                where 1: occupied and 0: free
            bearing : A 1 x n array of LIDAR bearings (numpy.ndarray)

            Returns
            -------
            d:                  Range
            coords:             Array of (x,y) coordinates
        """
        xr = np.array([xr])
        yr = np.array([yr])
        thetar = np.array([thetar])
        # print(f"Ray Tracing: {xr}, {yr}, {thetar}, {lAngle}")
        angle = np.array(thetar[:, None] + lAngle[None])
        # print(f"Angle: {angle}, {angle.shape}")
        x0 = np.array(xr/gridmap.xres)
        y0 = np.array(yr/gridmap.yres)

        # print(f"x0: {x0}, y0: {y0}")
        # print(f"{x0[:, None]}, {y0[:, None]}")
        # print(f"{[1, angle.shape[1]]}")
        x0 = np.tile(x0[:, None], [1, angle.shape[1]])
        y0 = np.tile(y0[:, None], [1, angle.shape[1]])
        assert angle.shape == x0.shape
        assert angle.shape == y0.shape

        # print(f"Final, x0: {x0}, y0: {y0}, angle: {angle}")
        def inCollision(x, y):
            return gridmap.inCollision(np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), True)

        (m,n) = gridmap.getShape()
        in_collision = inCollision(x0, y0)

        x0[x0 == np.floor(x0)] += 0.001
        y0[y0 == np.floor(y0)] += 0.001
        eps = 0.0001

        def inbounds(x, low, high):
            # return x in [low, high)
            return (x < high) * (x >= low)

        # Intersection with horizontal lines
        x = x0.copy()
        y = y0.copy()
        dir = np.tan(angle)
        xh = np.zeros_like(x)
        yh = np.zeros_like(y)
        foundh = np.zeros(x.shape, dtype=bool)
        seps = np.sign(np.cos(angle)) * eps
        while np.any(inbounds(x, 1, n)) and not np.all(foundh):
            x = np.where(seps > 0, np.floor(x+1), np.ceil(x-1))
            y = y0 + dir*(x-x0)
            inds = inCollision(x+seps, y) * np.logical_not(foundh) * inbounds(y, 0, m)
            if np.any(inds):
                xh[inds] = x[inds]
                yh[inds] = y[inds]
                foundh[inds] = True

        # Intersection with vertical lines
        x = x0.copy()
        y = y0.copy()
        eps = 1e-6
        dir = 1. / (np.tan(angle) + eps)
        xv = np.zeros_like(x)
        yv = np.zeros_like(y)
        foundv = np.zeros(x.shape, dtype=bool)
        seps = np.sign(np.sin(angle)) * eps
        while np.any(inbounds(y, 1, m)) and not np.all(foundv):
            y = np.where(seps > 0, np.floor(y+1), np.ceil(y-1))
            x = x0 + dir*(y-y0)
            inds = inCollision(x,y+seps) * np.logical_not(foundv) * inbounds(x, 0, n)
            if np.any(inds):
                xv[inds] = x[inds]
                yv[inds] = y[inds]
                foundv[inds] = True

        if not np.all(foundh + foundv):
            assert False, 'rayTracing: Error finding return'

        # account for poses in collision
        xh[in_collision] = x0[in_collision]
        yh[in_collision] = y0[in_collision]

        # get dist and coords
        dh = np.square(xh - x0) + np.square(yh - y0) + 1e7 * np.logical_not(foundh)
        dv = np.square(xv - x0) + np.square(yv - y0) + 1e7 * np.logical_not(foundv)
        d = np.where(dh < dv, dh, dv)
        cx = np.where(dh < dv, xh, xv)
        cy = np.where(dh < dv, yh, yv)
        coords = np.stack([cx, cy], axis=-1)
        return np.sqrt(d), coords

    def batchRayTracing(self, xr, yr, thetar, angles, gridmap):
        """Perform ray tracing for multiple angles at once, with caching

            Args
            -------
            xr, yr, thetar:   The robot's pose
            angles:           Array of LIDAR angles (in the LIDAR reference frame)
            gridmap:          The map of the environment

            Returns
            -------
            ranges:           Array of expected ranges
        """
        # Convert pose to discrete cells for caching (quantize to reduce cache size)
        x_cell = round(xr / 0.1) * 0.1
        y_cell = round(yr / 0.1) * 0.1
        theta_cell = round(thetar / 0.1) * 0.1

        # Initialize ranges array
        ranges = np.zeros(len(angles))

        # Process each angle
        for i, angle in enumerate(angles):
            # Global angle
            global_angle = thetar + angle
            # Round angle for cache key
            global_angle_cell = round(global_angle / 0.1) * 0.1

            # Create cache key
            cache_key = (x_cell, y_cell, global_angle_cell)

            # Check if we have a cached result
            if cache_key in self.ray_cache:
                ranges[i] = self.ray_cache[cache_key]
                self.cache_hits += 1
            else:
                # Compute ray tracing if not in cache
                range_val, _ = self.rayTracing2(xr, yr, thetar, angle, gridmap)
                ranges[i] = range_val

                # Cache the result if cache isn't too large
                if len(self.ray_cache) < self.max_cache_size:
                    self.ray_cache[cache_key] = range_val
                self.cache_misses += 1

        return ranges

    def rayTracing2(self, xr, yr, thetar, lAngle, gridmap):
        """A vectorized implementation of ray tracing

            Args
            -------
            (xr, yr, thetar): The robot's pose
            lAngle:           The LIDAR angle (in the LIDAR reference frame)
            gridmap:          An instance of the Gridmap class

            Returns
            -------
            d:                Range
            coords:           Array of (x,y) coordinates
        """
        # Convert to arrays for consistent handling
        xr = np.array([xr])
        yr = np.array([yr])
        thetar = np.array([thetar])
        lAngle = np.array([lAngle])

        # Calculate global angle
        angle = thetar[:, None] + lAngle[None]

        # Convert to grid coordinates
        x0 = np.array(xr/gridmap.xres)
        y0 = np.array(yr/gridmap.yres)

        # Tile for batch processing
        x0 = np.tile(x0[:, None], [1, angle.shape[1]])
        y0 = np.tile(y0[:, None], [1, angle.shape[1]])

        # Check starting position for collision
        def inCollision(x, y):
            return gridmap.inCollision(np.floor(x).astype(np.int32),
                                       np.floor(y).astype(np.int32),
                                       True)

        (m, n) = gridmap.getShape()
        in_collision = inCollision(x0, y0)

        # Avoid boundary issues
        x0[x0 == np.floor(x0)] += 0.001
        y0[y0 == np.floor(y0)] += 0.001
        eps = 0.0001

        # Helper function to check if a point is in bounds
        def inbounds(x, low, high):
            return (x < high) * (x >= low)

        # Intersection with horizontal lines
        x = x0.copy()
        y = y0.copy()
        dir = np.tan(angle)
        xh = np.zeros_like(x)
        yh = np.zeros_like(y)
        foundh = np.zeros(x.shape, dtype=bool)
        seps = np.sign(np.cos(angle)) * eps

        # Maximum iterations to prevent infinite loops
        max_iter = 100
        iter_count = 0

        while np.any(inbounds(x, 1, n)) and not np.all(foundh) and iter_count < max_iter:
            x = np.where(seps > 0, np.floor(x+1), np.ceil(x-1))
            y = y0 + dir*(x-x0)
            inds = inCollision(x+seps, y) * np.logical_not(foundh) * inbounds(y, 0, m)
            if np.any(inds):
                xh[inds] = x[inds]
                yh[inds] = y[inds]
                foundh[inds] = True
            iter_count += 1

        # Intersection with vertical lines
        x = x0.copy()
        y = y0.copy()
        eps = 1e-6
        dir = 1. / (np.tan(angle) + eps)
        xv = np.zeros_like(x)
        yv = np.zeros_like(y)
        foundv = np.zeros(x.shape, dtype=bool)
        seps = np.sign(np.sin(angle)) * eps

        # Reset iteration counter
        iter_count = 0

        while np.any(inbounds(y, 1, m)) and not np.all(foundv) and iter_count < max_iter:
            y = np.where(seps > 0, np.floor(y+1), np.ceil(y-1))
            x = x0 + dir*(y-y0)
            inds = inCollision(x, y+seps) * np.logical_not(foundv) * inbounds(x, 0, n)
            if np.any(inds):
                xv[inds] = x[inds]
                yv[inds] = y[inds]
                foundv[inds] = True
            iter_count += 1

        # Handle error case gracefully
        if not np.all(foundh + foundv):
            # Default to max range if ray tracing fails
            d = np.array([[self.zMax]])
            coords = np.array([[[xr[0] + self.zMax * np.cos(thetar[0] + lAngle[0]),
                               yr[0] + self.zMax * np.sin(thetar[0] + lAngle[0])]]])
            return d, coords

        # Handle collisions
        xh[in_collision] = x0[in_collision]
        yh[in_collision] = y0[in_collision]

        # Calculate distances
        dh = np.square(xh - x0) + np.square(yh - y0) + 1e7 * np.logical_not(foundh)
        dv = np.square(xv - x0) + np.square(yv - y0) + 1e7 * np.logical_not(foundv)

        # Select closest intersection
        d = np.where(dh < dv, dh, dv)
        cx = np.where(dh < dv, xh, xv)
        cy = np.where(dh < dv, yh, yv)

        # Convert back to world coordinates
        cx = cx * gridmap.xres
        cy = cy * gridmap.yres
        d = np.sqrt(d) * gridmap.xres  # Scale distance back to world units

        coords = np.stack([cx, cy], axis=-1)
        return d[0, 0], coords