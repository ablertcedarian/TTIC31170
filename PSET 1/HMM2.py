import numpy as np
import tqdm
import matplotlib.pyplot as plt

class HMM2(object):
    """A class for implementing HMMs.

    Attributes
    ----------
    envShape : list
        A two element list specifying the shape of the environment
    states : list
        A list of states specified by their (x, y) coordinates
    observations : list
        A list specifying the sequence of observations
    T : numpy.ndarray
        An N x N array encoding the transition probabilities, where
        T[i,j] is the probability of transitioning from state i to state j.
        N is the total number of states (envShape[0]*envShape[1])
    M : numpy.ndarray
        An M x N array encoding the emission probabilities, where
        M[k,i] is the probability of observing k from state i.
    pi : numpy.ndarray
        An N x 1 array encoding the prior probabilities

    Methods
    -------
    train(observations)
        Estimates the HMM parameters using a set of observation sequences
    viterbi(observations)
        Implements the Viterbi algorithm on a given observation sequence
    setParams(T, M, pi)
        Sets the transition (T), emission (M), and prior (pi) distributions
    getParams
        Queries the transition (T), emission (M), and prior (pi) distributions
    sub2ind(i, j)
        Convert integer (i,j) coordinates to linear index.
    """

    def __init__(self, envShape, T=None, M=None, pi=None):
        """Initialize the class.

        Attributes
        ----------
        envShape : list
            A two element list specifying the shape of the environment
        T : numpy.ndarray, optional
            An N x N array encoding the transition probabilities, where
            T[i,j] is the probability of transitioning from state i to state j.
            N is the total number of states (envShape[0]*envShape[1])
        M : numpy.ndarray, optional
            An M x N array encoding the emission probabilities, where
            M[k,i] is the probability of observing k from state i.
        pi : numpy.ndarray, optional
            An N x 1 array encoding the prior probabilities
        """
        self.envShape = envShape
        self.numStates = envShape[0] * envShape[1]

        if T is None:
            # Initial estimate of the transition function
            # where T[sub2ind(i',j'), sub2ind(i,j)] is the likelihood
            # of transitioning from (i,j) --> (i',j')
            self.T = np.zeros((self.numStates, self.numStates))

            # Self-transitions
            for i in range(self.numStates):
                self.T[i, i] = 0.2

            # Black rooms
            self.T[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
            self.T[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
            self.T[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
            self.T[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0

            # (1, 0) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(1, 0)] = 0.8

            # (2, 0) -->
            self.T[self.sub2ind(1, 0), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(2, 1), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(3, 0), self.sub2ind(2, 0)] = 0.8/3.0

            # (3, 0) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(3, 0)] = 0.8/2.0
            self.T[self.sub2ind(3, 1), self.sub2ind(3, 0)] = 0.8/2.0

            # (0, 1) --> (0, 2)
            self.T[self.sub2ind(0, 2), self.sub2ind(0, 1)] = 0.8

            # (2, 1) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(3, 1), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 1)] = 0.8/3.0

            # (3, 1) -->
            self.T[self.sub2ind(2, 1), self.sub2ind(3, 1)] = 0.8/2.0
            self.T[self.sub2ind(3, 0), self.sub2ind(3, 1)] = 0.8/2.0

            # (0, 2) -->
            self.T[self.sub2ind(0, 1), self.sub2ind(0, 2)] = 0.8/2.0
            self.T[self.sub2ind(1, 2), self.sub2ind(0, 2)] = 0.8/2.0

            # (1, 2) -->
            self.T[self.sub2ind(0, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(1, 3), self.sub2ind(1, 2)] = 0.8/3.0

            # (2, 2) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(2, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 1), self.sub2ind(2, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 3), self.sub2ind(2, 2)] = 0.8/3.0

            # (1, 3) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(1, 3)] = 0.8/2.0
            self.T[self.sub2ind(2, 3), self.sub2ind(1, 3)] = 0.8/2.0

            # (2, 3) -->
            self.T[self.sub2ind(1, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(3, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 3)] = 0.8/3.0

            # (3, 3) --> (2, 3)
            self.T[self.sub2ind(2, 3), self.sub2ind(3, 3)] = 0.8
        else:
            self.T = T

        if M is None:
            # Initial estimates of emission likelihoods, where
            # M[k, sub2ind(i,j)]: likelihood of observation k from state (i, j)
            self.M = np.ones((4, 16)) * 0.1

            # Black states
            self.M[:, self.sub2ind(0, 0)] = 0.25
            self.M[:, self.sub2ind(1, 1)] = 0.25
            self.M[:, self.sub2ind(0, 3)] = 0.25
            self.M[:, self.sub2ind(3, 2)] = 0.25

            self.M[self.obs2ind('r'), self.sub2ind(0, 1)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(0, 2)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(1, 0)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(1, 2)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(1, 3)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 0)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(2, 1)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(2, 2)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 3)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 0)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(3, 1)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 3)] = 0.7
        else:
            self.M = M

        if pi is None:
            # Initialize estimates of prior probabilities where
            # pi[(i, j)] is the likelihood of starting in state (i, j)
            self.pi = np.ones((16, 1))/12
            self.pi[self.sub2ind(0, 0)] = 0.0
            self.pi[self.sub2ind(1, 1)] = 0.0
            self.pi[self.sub2ind(0, 3)] = 0.0
            self.pi[self.sub2ind(3, 2)] = 0.0
        else:
            self.pi = pi

    def setParams(self, T, M, pi):
        """Set the transition, emission, and prior probabilities."""
        self.T = T
        self.M = M
        self.pi = pi

    def getParams(self):
        """Get the transition, emission, and prior probabilities."""
        return (self.T, self.M, self.pi)

    # Compute log-sum-exp in a numerically stable way
    def _log_sum_exp(self, log_probs):
        """Compute log(sum(exp(log_probs))) in a numerically stable way."""
        max_val = np.max(log_probs)
        return max_val + np.log(np.sum(np.exp(log_probs - max_val)))

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self, observations, states):
        """Estimate HMM parameters from training data via Baum-Welch.

        Parameters
        ----------
        observations : list
            A list specifying a set of observation sequences
            where observations[i] denotes a distinct sequence
        """
        # This function should set self.T, self.M, and self.pi
        print("Beginning Train!")

        total_log_likelihoods = [0]
        epsilon = 1e-3

        # Precompute log probabilities for efficiency
        log_T = np.log(self.T + 1e-10)  # Adding small epsilon to avoid log(0)
        log_M = np.log(self.M + 1e-10)
        log_pi = np.log(self.pi + 1e-10)

        try:
            for epoch in tqdm.tqdm(range(500), desc='Training'):
                log_alpha_seqs = []
                log_beta_seqs = []
                gamma_seqs = []
                xi_seqs = []

                for q, observation_sequence in enumerate(tqdm.tqdm(observations, desc='Observation Sequences Loop')):

                    # Forward Pass with log probabilities
                    log_alpha_seq = []

                    for t, observation in enumerate(observation_sequence):
                        log_alpha = np.full(self.numStates, -np.inf)  # Initialize with log(0)
                        obs_ind = self.obs2ind(observation)

                        if t == 0:
                            # Initial step: log(pi * M)
                            for i in range(self.envShape[0]):
                                for j in range(self.envShape[1]):
                                    lin_ind = self.sub2ind(i, j)
                                    log_alpha[lin_ind] = log_pi[lin_ind] + log_M[obs_ind, lin_ind]
                        else:
                            # Recursive step
                            for i in range(self.envShape[0]):
                                for j in range(self.envShape[1]):
                                    lin_ind = self.sub2ind(i, j)

                                    # Calculate log(sum(alpha[t-1] * T))
                                    log_sum_terms = []
                                    for prev_i in range(self.envShape[0]):
                                        for prev_j in range(self.envShape[1]):
                                            prev_lin_ind = self.sub2ind(prev_i, prev_j)
                                            log_sum_terms.append(log_alpha_seq[-1][prev_lin_ind] + log_T[lin_ind, prev_lin_ind])

                                    # Use log-sum-exp trick for numerical stability
                                    log_sum = self._log_sum_exp(np.array(log_sum_terms))
                                    log_alpha[lin_ind] = log_sum + log_M[obs_ind, lin_ind]

                        log_alpha_seq.append(log_alpha)

                    log_alpha_seqs.append(log_alpha_seq)

                    # Backward Pass with log probabilities
                    log_beta_seq = [np.zeros(self.numStates)]  # Initialize last beta as log(1) = 0

                    # Backwards Pass
                    for t in range(len(observation_sequence)-2, -1, -1):
                        log_beta = np.full(self.numStates, -np.inf)  # Initialize with log(0)
                        next_obs_ind = self.obs2ind(observation_sequence[t+1])

                        for i in range(self.envShape[0]):
                            for j in range(self.envShape[1]):
                                lin_ind = self.sub2ind(i, j)

                                # Calculate log(sum(T * M * beta[t+1]))
                                log_sum_terms = []
                                for next_i in range(self.envShape[0]):
                                    for next_j in range(self.envShape[1]):
                                        next_lin_ind = self.sub2ind(next_i, next_j)
                                        log_sum_terms.append(log_T[next_lin_ind, lin_ind] +
                                                            log_M[next_obs_ind, next_lin_ind] +
                                                            log_beta_seq[0][next_lin_ind])

                                log_sum = self._log_sum_exp(np.array(log_sum_terms))
                                log_beta[lin_ind] = log_sum

                        log_beta_seq.insert(0, log_beta)  # Insert at beginning

                    log_beta_seqs.append(log_beta_seq)

                    # Calculate gamma using log domain
                    gamma_seq = []
                    for t in range(len(observations[0])):

                        # Calculate log(gamma) = log(alpha) + log(beta) - log(sum(alpha*beta))
                        log_gamma = log_alpha_seq[t] + log_beta_seq[t]
                        # Normalize using log-sum-exp
                        log_norm = self._log_sum_exp(log_gamma)
                        log_gamma = log_gamma - log_norm

                        # Convert back to probability space for the rest of the algorithm
                        gamma_t = np.exp(log_gamma)
                        gamma_seq.append(gamma_t)

                    gamma_seqs.append(gamma_seq)

                    # Calculate xi using log domain
                    xi_seq = []
                    for t in range(len(observations[0]) - 1):
                        next_obs_ind = self.obs2ind(observation_sequence[t+1])

                        xi_t = np.zeros((self.numStates, self.numStates))
                        log_xi_normalizer = -np.inf

                        # First calculate unnormalized xi in log domain
                        log_xi_t = np.full((self.numStates, self.numStates), -np.inf)

                        for curr_i in range(self.envShape[0]):
                            for curr_j in range(self.envShape[1]):
                                curr_lin_ind = self.sub2ind(curr_i, curr_j)

                                for next_i in range(self.envShape[0]):
                                    for next_j in range(self.envShape[1]):
                                        next_lin_ind = self.sub2ind(next_i, next_j)

                                        log_xi_t[curr_lin_ind, next_lin_ind] = (
                                            log_alpha_seq[t][curr_lin_ind] +
                                            log_T[next_lin_ind, curr_lin_ind] +
                                            log_M[next_obs_ind, next_lin_ind] +
                                            log_beta_seq[t+1][next_lin_ind]
                                        )

                        # Flatten for normalization
                        flat_log_xi = log_xi_t.flatten()
                        log_xi_normalizer = self._log_sum_exp(flat_log_xi)

                        # Normalize and convert back to probability space
                        for curr_i in range(self.envShape[0]):
                            for curr_j in range(self.envShape[1]):
                                curr_lin_ind = self.sub2ind(curr_i, curr_j)

                                for next_i in range(self.envShape[0]):
                                    for next_j in range(self.envShape[1]):
                                        next_lin_ind = self.sub2ind(next_i, next_j)

                                        xi_t[curr_lin_ind, next_lin_ind] = np.exp(
                                            log_xi_t[curr_lin_ind, next_lin_ind] - log_xi_normalizer
                                        )

                        xi_seq.append(xi_t)

                    xi_seqs.append(xi_seq)

                # The rest of the update logic remains the same
                print("Setting pi!")
                pi_new = []
                for row in range(self.envShape[0]):
                    for col in range(self.envShape[1]):
                        lin_ind = self.sub2ind(row, col)

                        pi_new.append(
                            np.average([
                                gamma_seqs[q][0][lin_ind]
                                for q in range(len(observations))
                            ])
                        )
                        if pi_new[-1] > 1:
                            print(f"    {row,col}: {self.pi[lin_ind]} to {pi_new[-1]}")
                self.pi = pi_new

                print("Setting T!")
                t_new = np.zeros((self.numStates, self.numStates))
                for row in range(self.envShape[0]):
                    for col in range(self.envShape[1]):
                        lin_ind = self.sub2ind(row, col)

                        for next_row in range(self.envShape[0]):
                            for next_col in range(self.envShape[1]):
                                next_lin_ind = self.sub2ind(next_row, next_col)

                                num_transitions_from_i_to_j = np.sum([
                                    np.sum([
                                        xi_seqs[q][t][lin_ind, next_lin_ind]
                                        for t in range(len(observations[0]) - 1)
                                    ])
                                    for q in range(len(observations))
                                ])
                                num_transitions_from_i = np.sum([
                                    np.sum([
                                        gamma_seqs[q][t][lin_ind]
                                        for t in range(len(observations[0]) - 1)
                                    ])
                                    for q in range(len(observations))
                                ])
                                t_new[next_lin_ind, lin_ind] = num_transitions_from_i_to_j / num_transitions_from_i if num_transitions_from_i > 0 else 0

                                if t_new[next_lin_ind, lin_ind] > 1:
                                    print(f"    {row,col} -> {next_row, next_col}: {self.T[next_lin_ind, lin_ind]} -> {t_new[next_lin_ind, lin_ind]}")
                self.T = t_new

                # Reset black room probabilities
                self.T[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
                self.T[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
                self.T[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
                self.T[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0

                print("Setting M!")
                m_new = np.ones((4, 16))
                for row in range(self.envShape[0]):
                    for col in range(self.envShape[1]):
                        lin_ind = self.sub2ind(row, col)

                        for obs in ['r', 'g', 'b', 'y']:
                            obs_ind = self.obs2ind(obs)

                            num_times_state_i_observe_m = np.sum([
                                np.sum([
                                    gamma_seqs[q][t][lin_ind]
                                    if self.obs2ind(observations[q][t]) == obs_ind
                                    else 0
                                    for t in range(len(observations[q]))
                                ])
                                for q in range(len(observations))
                            ])

                            num_times_state_i = np.sum([
                                np.sum([
                                    gamma_seqs[q][t][lin_ind]
                                    for t in range(len(observations[q]))
                                ])
                                for q in range(len(observations))
                            ])

                            m_new[obs_ind, lin_ind] = num_times_state_i_observe_m / num_times_state_i if num_times_state_i > 0 else 0

                            if m_new[obs_ind, lin_ind] > 1:
                                print(f"    {obs_ind} @ {row, col}: {self.M[obs_ind, lin_ind]} -> {m_new[obs_ind, lin_ind], num_times_state_i_observe_m, num_times_state_i}")
                self.M = m_new

                # Reset black room probabilities ?
                self.M[:, self.sub2ind(0, 0)] = 0.25
                self.M[:, self.sub2ind(1, 1)] = 0.25
                self.M[:, self.sub2ind(0, 3)] = 0.25
                self.M[:, self.sub2ind(3, 2)] = 0.25

                # Calculate log-likelihood from forward variables
                sequence_log_likelihoods = []
                for q in range(len(observations)):
                    log_likelihood = self._log_sum_exp(log_alpha_seqs[q][-1])
                    sequence_log_likelihoods.append(log_likelihood)

                total_log_likelihood = np.sum(sequence_log_likelihoods)
                total_log_likelihoods.append(total_log_likelihood)

                print(f"Epoch {epoch}, Total log-likelihood: {total_log_likelihood}")
                # print(total_log_likelihoods)

                # Update log probability arrays for next epoch
                log_T = np.log(self.T + 1e-10)
                log_M = np.log(self.M + 1e-10)
                log_pi = np.log(np.array(self.pi).reshape(-1, 1) + 1e-10)

                if epoch > 0 and abs(total_log_likelihoods[-1] - total_log_likelihoods[-2]) < epsilon:
                    break
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        finally:
            # Always save log-likelihoods, even on Ctrl+C
            np.savetxt("ttls.txt", total_log_likelihoods, fmt="%.10f")

            log_likelihoods = np.loadtxt("ttls.txt")
            plt.figure(figsize=(8, 5))
            plt.plot(log_likelihoods, marker='o')
            plt.title("Log-Likelihood vs Training Epoch")
            plt.xlabel("Epoch")
            plt.yscale('log')
            plt.ylabel("Total Log-Likelihood")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("log_likelihood_curve.png")
            plt.show()

    def viterbi(self, observations):
        """Implement the Viterbi algorithm in log domain.

        Parameters
        ----------
        observations : list
            A list specifying the sequence of observations, where each observation is a string (e.g., 'r')

        Returns
        -------
        states : list
            List of predicted sequence of states, each specified as (x, y) pair
        """
        # Precompute log probabilities for efficiency
        log_T = np.log(self.T + 1e-10)
        log_M = np.log(self.M + 1e-10)
        log_pi = np.log(np.array(self.pi) + 1e-10)

        # Forward Pass
        log_delta_seq = []
        pre_seq = []

        for t, observation in enumerate(observations):
            obs_ind = self.obs2ind(observation)

            if t == 0:
                # Initial step: log(pi * M)
                log_delta_seq.append([
                    log_M[obs_ind, lin_index] + log_pi[lin_index]
                    for lin_index in range(self.numStates)
                ])
                pre_seq.append([None] * self.numStates)
                continue

            log_delta_t = []
            pre_t = []

            for lin_index in range(self.numStates):
                best_log_val = -np.inf
                best_prev = -1

                for prev_lin_index in range(self.numStates):
                    log_val = log_T[lin_index, prev_lin_index] + log_delta_seq[t-1][prev_lin_index]
                    if log_val > best_log_val:
                        best_log_val = log_val
                        best_prev = prev_lin_index

                log_delta_val = log_M[obs_ind, lin_index] + best_log_val
                log_delta_t.append(log_delta_val)
                pre_t.append(best_prev)

            log_delta_seq.append(log_delta_t)
            pre_seq.append(pre_t)

        # Select most likely terminal state
        x_seq = np.empty(len(observations))
        x_seq[:] = np.nan
        x_seq[len(observations) - 1] = np.argmax(log_delta_seq[-1])

        # Backward Pass
        for t in range(len(observations) - 2, -1, -1):
            x_seq[t] = pre_seq[t+1][int(x_seq[t+1])]

        x_seq_new = [
            self.ind2sub(int(x))
            for x in x_seq
        ]

        return x_seq_new

    def forward(self, alpha, z):
        """Implement one forward step."""
        pass

    def computeGamma(self, alpha, beta, norm):
        """Compute P(X[t] | Z^T)."""
        # CODE GOES HERE
        pass

    def computeXis(self, alpha, beta, z):
        """Compute xi as an array comprised of each xi-xj pair."""
        pass

    def getLogStartProb(self, state):
        """Return the log probability of a particular state."""
        return np.log(self.pi[state])

    def getLogTransProb(self, fromState, toState):
        """Return the log probability associated with a state transition."""
        return np.log(self.T[toState, fromState])

    def getLogOutputProb(self, state, output):
        """Return the log probability of a state-dependent observation."""
        return np.log(self.M[output, state])

    def sub2ind(self, i, j):
        """Convert subscript (i,j) to linear index."""
        return (self.envShape[1]*i + j)

    def ind2sub(self, lin_index):
        return (lin_index // self.envShape[1], lin_index % self.envShape[1])

    def obs2ind(self, obs):
        """Convert observation string to linear index."""
        obsToInt = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
        return obsToInt[obs]