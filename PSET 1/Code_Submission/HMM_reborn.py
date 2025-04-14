import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle

class HMM_reborn(object):
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
            T[j, i] is the probability of transitioning from state i to state j.
            N is the total number of states (envShape[0]*envShape[1])
        M : numpy.ndarray, optional
            An M x N array encoding the emission probabilities, where
            M[k,i] is the probability of observing k from state i.
        pi : numpy.ndarray, optional
            An N x 1 array encoding the prior probabilities
        """
        self.envShape = envShape
        self.numStates = envShape[0] * envShape[1]
        self.blackroom_indices = [
            self.sub2ind(0, 0),
            self.sub2ind(1, 1),
            self.sub2ind(0, 3),
            self.sub2ind(3, 2),
        ]

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

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self, observations, states, printer=False):
        """Estimate HMM parameters from training data via Baum-Welch.

        Parameters
        ----------
        observations : list
            A list specifying a set of observation sequences
            where observations[i] denotes a distinct sequence
        """
        # This function should set self.T, self.M, and self.pi

        lls = []
        try:
            for iter_num in range(50):
                print(f"\n--- Iteration {iter_num + 1} ---")

                gamma_seqs = []
                xi_seqs = []
                c_seqs = []

                for observation_seq_index, observation_seq in enumerate(observations):
                    alpha_norm_seq = []
                    c_seq = []
                    beta_norm_seq = [[] for _ in range(len(observation_seq))]

                    if printer and observation_seq_index == 0: print(f"\n-- Sequence {observation_seq_index + 1} --")

                    # Forward Pass
                    for t, observation in enumerate(observation_seq):
                        alpha_t = self.forward(alpha_norm_seq, observation_seq, t)
                        c_t = np.sum(alpha_t)
                        alpha_norm_t = alpha_t / c_t
                        alpha_norm_seq.append(alpha_norm_t)
                        c_seq.append(c_t)
                        if printer and observation_seq_index == 0: print(f"  Alpha at t={t}: Shape={alpha_t.shape}, First 5 values={alpha_t[:5]}")

                    # Backward Pass
                    for t in range(len(observation_seq) - 1, -1, -1):
                        beta_t = self.backward(beta_norm_seq, observation_seq, t)
                        beta_norm_t = beta_t / c_seq[t + 1] if t < len(observation_seq) - 1 else beta_t
                        beta_norm_seq[t] = beta_norm_t
                        if printer and observation_seq_index == 0: print(f"  Beta at t={t}: Shape={beta_t.shape}, First 5 values={beta_t[:5]}")


                    # Temporary Variables (Gamma and Xi)
                    gamma_seq = []
                    xi_seq = []

                    # Forward Pass
                    for t, observation in enumerate(observation_seq):
                        gamma_t = self.computeGamma(alpha_norm_seq, beta_norm_seq, t)
                        gamma_seq.append(gamma_t)
                        if printer and observation_seq_index == 0: print(f"  Gamma at t={t}: Shape=({len(gamma_t)}), First 5 values={gamma_t[:5]}")


                    # Forward Pass
                    for t, observation in enumerate(observation_seq[:-1]): # exclude the last element
                        xi_t = self.computeXis(alpha_norm_seq, beta_norm_seq, observation_seq, t)
                        xi_seq.append(xi_t)
                        if printer and observation_seq_index == 0: print(f"  Xi at t={t}: Shape={xi_t.shape}, First 9 elements=\n{xi_t[:3, :3]}")

                    gamma_seqs.append(gamma_seq)
                    xi_seqs.append(xi_seq)
                    c_seqs.append(c_seq)

                ## Update Parameters ##

                # Update Pi
                for i in range(self.numStates):
                    self.pi[i] = np.sum([
                        gamma_seqs[q][0][i]
                        for q in range(len(observations))
                    ]) / len(observations)
                if printer: print(f"\nFinal Pi after iteration {iter_num + 1}:\n{self.pi.flatten()}")

                # Update T
                for i in range(self.numStates):
                    if i in self.blackroom_indices:
                        continue
                    # Save some computation
                    norm = np.sum([
                        np.sum([
                            gamma_seqs[q][t][i]
                            for t in range(len(observations[q]) - 1)
                        ])
                        for q in range(len(observations))
                    ])

                    if norm > 0:
                        for j in range(self.numStates):
                            if j in self.blackroom_indices:
                                continue
                            self.T[j, i] = (
                                np.sum([
                                    np.sum([
                                        xi_seqs[q][t][i, j]
                                        for t in range(len(observations[q]) - 1)
                                    ])
                                    for q in range(len(observations))
                                ])
                            ) / norm
                    else:
                        print(f"Warning: Normalization factor for T at state {i} is zero.")
                if printer: print(f"\nFinal T after iteration {iter_num + 1}:\n{self.T}")


                for i in self.blackroom_indices:
                    self.T[:, i] = 0.0
                    self.T[i, :] = 0.0
                    self.T[i, i] = 1.0

                # Update M
                for i in range(self.numStates):
                    if i in self.blackroom_indices:
                        continue

                    norm = np.sum([
                        np.sum([
                            gamma_seqs[q][t][i]
                            for t in range(len(observations[q]))
                        ])
                        for q in range(len(observations))
                    ])

                    if norm > 0:
                        for obs_ind in range(4):
                            self.M[obs_ind, i] = (
                                np.sum([
                                    np.sum([
                                        gamma_seqs[q][t][i]
                                        if self.obs2ind(observations[q][t]) == obs_ind
                                        else 0
                                        for t in range(len(observations[q]))
                                    ])
                                    for q in range(len(observations))
                                ])
                            ) / norm
                    else:
                        print(f"Warning: Normalization factor for M at state {i} is zero.")
                if printer: print(f"\nFinal M after iteration {iter_num + 1}:\n{self.M}")

                # for i in self.blackroom_indices:
                #     self.M[:, i] = np.array([0.25, 0.25, 0.25, 0.25])


                # Calculate Log Likelihood
                total_likelihood = - np.sum([
                    np.sum([
                        np.log(c_seqs[q][t])
                        for t in range(len(c_seqs[q]))
                    ])
                    for q in range(len(c_seqs))
                ])
                lls.append(total_likelihood)

                print(f"Log Likelihood: ", total_likelihood)

            print("\n--- Final Parameters ---")
            print(f"Final Pi:\n{self.pi.flatten()}")
            print(f"Final T:\n{self.T}")
            print(f"Final M:\n{self.M}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        finally:
            # Always save log-likelihoods, even on Ctrl+C
            np.savetxt("ttls.txt", lls, fmt="%.10f")

            log_likelihoods = np.loadtxt("ttls.txt")
            plt.figure(figsize=(8, 5))
            plt.plot(log_likelihoods, marker='o')
            plt.title("Log-Likelihood vs Training Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Total Log-Likelihood")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("log_likelihood_curve.png")
            plt.show()

            fileName = f"trained-model_epochs_{iter_num}.pkl"
            print("Saving trained model as " + fileName)
            pickle.dump({'T': self.T, 'M': self.M, 'pi': self.pi}, open(fileName, "wb"))




    def viterbi2(self, observations):
        """Implement the Viterbi algorithm in log space."""

    def viterbi(self, observations):
        """Implement the Viterbi algorithm.

        Parameters
        ----------
        observations : list
            A list specifying the sequence of observations, where each o
            observation is a string (e.g., 'r')

        Returns
        -------
        states : list
            List of predicted sequence of states, each specified as (x, y) pair
        """
        # CODE GOES HERE
        # Return the list of predicted states, each specified as (x, y) pair

        prob = np.zeros((len(observations), self.numStates))
        prev = np.zeros((len(observations), self.numStates))
        for i in range(self.numStates):
            prob[0][i] = self.pi[i] * self.M[self.obs2ind(observations[0]), i]

        for t in range(1, len(observations)):
            for i in range(self.numStates):
                for j in range(self.numStates):
                    new_prob = prob[t - 1][j] * self.T[j, i] * self.M[self.obs2ind(observations[t]), i]
                    if new_prob > prob[t][i]:
                        prob[t][i] = new_prob
                        prev[t][i] = j

        path = np.zeros(len(observations), dtype=int)
        path[-1] = np.argmax(prob[-1])
        # print(prob[-1], path[-1])

        for t in range(len(observations) - 2, -1, -1):
            path[t] = prev[t + 1][path[t + 1]]

        return [self.ind2sub(path[i]) for i in range(len(observations))]


    def forward(self, alpha_seq, observation_seq, t):
        """Implement one forward step."""
        alpha_t = np.zeros(self.numStates)
        z = observation_seq[t]

        for i in range(self.numStates):
            if i in self.blackroom_indices:
                alpha_t[i] = 0.0
                continue

            if t == 0:
                alpha_t[i] = self.M[self.obs2ind(z), i] * self.pi[i]
            else:
                alpha_t[i] = self.M[self.obs2ind(z), i] * (
                    np.sum([
                        self.T[i, j] * alpha_seq[-1][j]
                        for j in range(self.numStates)
                    ])
                )

        return alpha_t

    def backward(self, beta_seq, observation_seq, t):
        """Implement one backward step."""
        beta_t = np.zeros(self.numStates)
        z_next = observation_seq[t + 1] if t < len(observation_seq) - 1 else None

        for i in range(self.numStates):
            if i in self.blackroom_indices:
                beta_t[i] = 0.0
                continue

            if t == len(beta_seq) - 1:
                beta_t[i] = 1.0 if i not in self.blackroom_indices else 0
            else:
                beta_t[i] = np.sum([
                    self.M[self.obs2ind(observation_seq[t + 1]), j] * self.T[j, i] * beta_seq[t + 1][j]
                    for j in range(self.numStates)
                ])

        return beta_t

    def computeGamma(self, alpha_seq, beta_seq, t):
        """Compute P(X[t] | Z^T)."""
        gamma_t = []
        norm = np.sum([
            alpha_seq[t][j] * beta_seq[t][j]
            for j in range(self.numStates)
        ])

        for i in range(self.numStates):
            gamma_t.append(
                (alpha_seq[t][i] * beta_seq[t][i])
                / norm
            )
            if i in self.blackroom_indices:
                assert gamma_t[-1] == 0, f"Gamma for blackroom {i} at {t} is not 0, instead {gamma_t[-1]} bc {alpha_seq[t][i], beta_seq[t][i], norm}"

        return gamma_t

    def computeXis(self, alpha_seq, beta_seq, observation_seq, t):
        """Compute xi as an array comprised of each xi-xj pair."""
        xi_t = np.zeros((self.numStates, self.numStates))
        norm = np.sum([
            np.sum([
                (
                    alpha_seq[t][k]
                    * self.T[w, k]
                    * self.M[self.obs2ind(observation_seq[t + 1]), w]
                    * beta_seq[t + 1][w]
                )
                for w in range(self.numStates)
            ])
            for k in range(self.numStates)
        ])

        for i in range(self.numStates):
            if i in self.blackroom_indices:
                xi_t[i, :] = 0.0
                continue

            for j in range(self.numStates):
                if j in self.blackroom_indices:
                    xi_t[i, j] = 0.0
                    continue

                xi_t[i, j] = (
                    alpha_seq[t][i]
                    * self.T[j, i]
                    * self.M[self.obs2ind(observation_seq[t + 1]), j]
                    * beta_seq[t + 1][j]
                ) / norm
        return xi_t

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
