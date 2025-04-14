import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle

class HMM(object):
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
    def train(self, observations, states):
        """Estimate HMM parameters from training data via Baum-Welch.

        Parameters
        ----------
        observations : list
            A list specifying a set of observation sequences
            where observations[i] denotes a distinct sequence
        """
        # This function should set self.T, self.M, and self.pi

        # Check transition matrix (T)
        def check_transition_matrix(T, numStates):
            # Check values are between 0 and 1
            assert np.all((T >= 0) & (T <= 1)), "Transition matrix contains values outside [0,1]"

            # Check that every column sums to 1 (assuming your convention is T[j,i] is prob of i->j)
            for i in range(numStates):
                col_sum = np.sum(T[:, i])
                assert np.isclose(col_sum, 1.0, atol=1e-6) or np.isclose(col_sum, 0.0, atol=1e-6), \
                    f"Column {i} in transition matrix sums to {col_sum}, not 1 or 0"

        # Check emission matrix (M)
        def check_emission_matrix(M):
            # Check values are between 0 and 1
            assert np.all((M >= 0) & (M <= 1)), "Emission matrix contains values outside [0,1]"

            # Check that every column sums to 1
            for i in range(M.shape[1]):
                col_sum = np.sum(M[:, i])
                assert np.isclose(col_sum, 1.0, atol=1e-6), \
                    f"Column {i} in emission matrix sums to {col_sum}, not 1"

        # Check reasonable beta values
        def check_beta_values(beta_seq):
            for t, beta in enumerate(beta_seq):
                # Beta values should be between 0 and 1 when normalized
                assert np.all((np.min(beta) >= 0) & (np.max(beta) <= 1e10)), \
                    f"Beta at t={t} contains values outside [0,1]: min={np.min(beta)}, max={np.max(beta)}"

                # Optional: check for reasonable magnitudes
                if np.max(beta) > 1e10:
                    print(f"Warning: Large beta value detected at t={t}: max={np.max(beta)}")

        # Function to check alpha values
        def check_alpha_values(alpha_seq):
            for t, alpha in enumerate(alpha_seq):
                # Alpha should sum to 1 after normalization
                assert np.isclose(np.sum(alpha), 1.0, atol=1e-6), \
                    f"Alpha at t={t} sums to {np.sum(alpha)}, not 1"

                # Alpha values should be between 0 and 1
                assert np.all((alpha >= 0) & (alpha <= 1)), \
                    f"Alpha at t={t} contains values outside [0,1]"

        # Function to check gamma values
        def check_gamma_values(gamma_seq):
            for t, gamma in enumerate(gamma_seq):
                # Gamma should sum to 1 at each time step
                assert np.isclose(np.sum(gamma), 1.0, atol=1e-6), \
                    f"Gamma at t={t} sums to {np.sum(gamma)}, not 1"

                # Gamma values should be between 0 and 1
                assert np.all((np.min(gamma) >= 0) & (np.max(gamma) <= 1)), \
                    f"Gamma at t={t} contains values outside [0,1]"

        print("Beginning Train!")

        total_log_likelihoods = [0]
        epsilon = 1e-3

        black_room_indices = [
            self.sub2ind(0, 0),
            self.sub2ind(1, 1),
            self.sub2ind(0, 3),
            self.sub2ind(3, 2)
        ]

        try:
            for epoch in tqdm.tqdm(range(8), desc='Training'):
                alpha_seqs = []
                c_seqs = []
                beta_seqs = []
                gamma_seqs = []
                xi_seqs = []
                for q, observation_sequence in enumerate(tqdm.tqdm(observations, desc='Observation Sequences Loop')):

                    alpha_seq = []
                    c_seq = []
                    beta_seq = [[] for _ in range(len(observation_sequence))]
                    gamma_seq = []
                    xi_seq = []

                    # Forward Pass
                    for t, observation in enumerate(observation_sequence):

                        alpha = np.zeros(self.numStates)
                        for i in range(self.envShape[0]):
                            for j in range(self.envShape[1]):
                                lin_ind = self.sub2ind(i, j)

                                prev_pos = None
                                if t == 0:
                                    prev_pos = self.pi[lin_ind]
                                else:
                                    prev_pos = 0
                                    for prev_i in range(self.envShape[0]):
                                        for prev_j in range(self.envShape[1]):
                                            prev_lin_ind = self.sub2ind(prev_i, prev_j)
                                            prev_pos += alpha_seq[-1][prev_lin_ind] * self.T[lin_ind, prev_lin_ind]

                                alpha[lin_ind] = prev_pos * self.M[self.obs2ind(observation), lin_ind]

                        C_t = np.sum(alpha)
                        c_seq.append(C_t)
                        alpha /= C_t # Normalize
                        alpha_seq.append(alpha)

                    # if q == 1:
                        # print(alpha_seq)
                    check_alpha_values(alpha_seq)

                    alpha_seqs.append(alpha_seq)
                    c_seqs.append(c_seq)

                    beta_seq[-1] = [1 for _ in range(self.numStates)]

                    for idx in black_room_indices:
                        beta_seq[-1][idx] = 0

                    # Backwards Pass
                    for t in range(len(observation_sequence)-2, -1, -1):

                        beta = np.zeros(self.numStates)
                        for i in range(self.envShape[0]):
                            for j in range(self.envShape[1]):
                                lin_ind = self.sub2ind(i, j)

                                if lin_ind in black_room_indices:
                                    beta[lin_ind] = 0

                                prev_pos = 0
                                for prev_i in range(self.envShape[0]):
                                    for prev_j in range(self.envShape[1]):
                                        prev_lin_ind = self.sub2ind(prev_i, prev_j)
                                        # if ((t == 0) or (t == 100) or (t == 199)) and (i == 0 and j == 0 and prev_i == 0 and prev_j == 0):
                                        #     print(t, beta_seq[t+1], beta_seq[t+1][prev_lin_ind])

                                        if prev_lin_ind in black_room_indices:
                                            continue
                                        prev_pos += (
                                            beta_seq[t+1][prev_lin_ind]
                                            * self.T[prev_lin_ind, lin_ind]
                                            * self.M[self.obs2ind(observation_sequence[t+1]), prev_lin_ind]
                                        )
                                        # if prev_pos > 10:
                                        #     print(f"        {epoch, t}: {i,j}: beta_indiv {beta_seq[t+1][prev_lin_ind], self.T[prev_lin_ind, lin_ind], self.M[self.obs2ind(observation_sequence[t+1]), prev_lin_ind]}")

                                beta[lin_ind] = prev_pos
                                # if (beta[lin_ind] > 10**2):
                                if (beta[lin_ind] > 10**10) or (beta[lin_ind] == 0 and (lin_ind not in black_room_indices)):
                                # if (beta[lin_ind] / c_seq[t+1] > 10**2) or (beta[lin_ind] == 0):
                                    print(f"    in {epoch}, {t}, ({i,j}), beta: {beta[lin_ind] / c_seq[t+1], beta[lin_ind], c_seq[t+1]}")
                                    # raise Exception

                        # beta_seq[t] = beta
                        beta_seq[t] = beta / c_seq[t+1]
                        # beta_seq[t] = beta / c_seq[t]

                    # for t in range(len(observation_sequence) - 1):
                    #     beta_seq[t] = beta_seq[t] / c_seq[t + 1]
                    # check_beta_values(beta_seq)
                    assert np.all((beta_seq[t] >= 0) & (beta_seq[t] <= 1e10)), \
                        f"Beta at t={t} contains values outside [0,1]: min={np.min(beta_seq[t])}, max={np.max(beta_seq[t])}"

                    beta_seqs.append(beta_seq)

                    gamma_seq = []
                    for t in range(len(observations[0])):

                        gamma_t = []
                        norm_constant = np.sum(
                            [
                                alpha_seq[t][x] * beta_seq[t][x]
                                for x in range(self.numStates)
                            ]
                        )
                        for row in range(self.envShape[0]):
                            for col in range(self.envShape[1]):
                                lin_ind = self.sub2ind(row, col)

                                if lin_ind in black_room_indices:
                                    gamma_t.append(0)
                                else:
                                    gamma_t.append(
                                        (alpha_seq[t][lin_ind] * beta_seq[t][lin_ind]) / norm_constant
                                    )
                                    if (gamma_t[-1] > 1) or (norm_constant == 0) or (np.isnan(norm_constant) or alpha_seq[t][lin_ind] > 1 or beta_seq[t][lin_ind] > 10**10):
                                        print(f"    in {epoch}, {t} gamma: {gamma_t[-1], alpha_seq[t][lin_ind], beta_seq[t][lin_ind], norm_constant}")
                                        raise Exception

                        gamma_seq.append(gamma_t)

                    check_gamma_values(gamma_seq)

                    gamma_seqs.append(gamma_seq)
                    # print(len(gamma_seqs), len(gamma_seqs[0]), len(gamma_seqs[0][0]), len(gamma_seqs[0][0][0]))
                    # print(gamma_seq)
                    for t in range(len(observation_sequence)):
                        assert(np.isclose(np.sum(gamma_seq[t]), 1, atol=1e-2)), f"Normalized gamma_t sum at {q, t} is {np.sum(gamma_seq[t])} (expected 1)"
                        assert(len(gamma_seq[t]) == self.numStates), f"Gamma_t length at {q,t} is {len(gamma_seq[t])} (expected {self.numStates})"

                    xi_seq = []
                    for t in range(len(observations[0]) - 1):
                        norm_factor_t = (
                            np.sum([
                                np.sum([
                                    (
                                        alpha_seq[t][k_lin_ind]
                                        * beta_seq[t + 1][w_lin_ind]
                                        * self.T[w_lin_ind, k_lin_ind]
                                        * self.M[self.obs2ind(observation_sequence[t + 1]), w_lin_ind]
                                    )
                                    for w_lin_ind in range(self.numStates)
                                ])
                                for k_lin_ind in range(self.numStates)
                            ])
                        )

                        xi_t = np.zeros((self.numStates, self.numStates))
                        for curr_row in range(self.envShape[0]):
                            for curr_col in range(self.envShape[1]):
                                curr_lin_ind = self.sub2ind(curr_row, curr_col)

                                if curr_lin_ind in black_room_indices:
                                    xi_t[curr_lin_ind, :] = 0
                                else:
                                    for next_row in range(self.envShape[0]):
                                        for next_col in range(self.envShape[1]):
                                            next_lin_ind = self.sub2ind(next_row, next_col)

                                            if next_lin_ind in black_room_indices:
                                                xi_t[curr_lin_ind, next_lin_ind] = 0
                                            else:
                                                xi_t[curr_lin_ind, next_lin_ind] = (
                                                    alpha_seq[t][curr_lin_ind]
                                                    * self.T[next_lin_ind, curr_lin_ind]
                                                    * self.M[self.obs2ind(observation_sequence[t + 1]), next_lin_ind]
                                                    * beta_seq[t + 1][next_lin_ind]
                                                ) / norm_factor_t
                                                # if (xi_t[curr_lin_ind, next_lin_ind] > 1):
                                                #     print(f"    {curr_row, curr_col} -> {next_row, next_col}: {xi_t[curr_lin_ind, next_lin_ind], alpha_seq[t][curr_lin_ind] , self.T[curr_lin_ind, next_lin_ind], self.M[self.obs2ind(observation_sequence[t + 1]), next_lin_ind] , beta_seq[t + 1][next_lin_ind]} ")
                        # xi_seq.append(xi_t / np.sum(xi_t))
                        xi_seq.append(xi_t)
                        # assert np.isclose(np.sum(xi_t / np.sum(xi_t)), 1, atol=1e-6), f"Normalized xi_t sum is {np.sum(xi_t / np.sum(xi_t))} (expected 1)"

                        for curr_row in range(self.envShape[0]):
                            for curr_col in range(self.envShape[1]):
                                curr_lin_ind = self.sub2ind(curr_row, curr_col)

                                for next_row in range(self.envShape[0]):
                                    for next_col in range(self.envShape[1]):
                                        next_lin_ind = self.sub2ind(next_row, next_col)
                                        if xi_seq[-1][curr_lin_ind, next_lin_ind] > 1:
                                            print(f"    {curr_row, curr_col} -> {next_row, next_col}: xi {xi_t[curr_lin_ind, next_lin_ind], alpha_seq[t][curr_lin_ind] , self.T[curr_lin_ind, next_lin_ind], self.M[self.obs2ind(observation_sequence[t + 1]), next_lin_ind] , beta_seq[t + 1][next_lin_ind]} ")


                    xi_seqs.append(xi_seq)

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
                            print(f"    {row,col}: pi {self.pi[lin_ind]} to {pi_new[-1]}")
                self.pi = pi_new

                tol = 1e-3

                # Check that all entries are between 0 and 1.
                if not np.all((np.min(self.pi) >= 0) & (np.max(self.pi) <= 1)):
                    raise ValueError("Initial distribution pi contains entries outside the interval [0, 1].")

                # Check that the entries sum to 1.
                pi_sum = np.sum(self.pi)
                if not np.isclose(pi_sum, 1.0, atol=tol):
                    raise ValueError(f"Initial distribution pi sums to {pi_sum}, expected 1.")

                print("     Initial distribution pi is valid.")

                print("Setting T!")
                t_new = np.zeros((self.numStates, self.numStates))
                for row in range(self.envShape[0]):
                    for col in range(self.envShape[1]):
                        lin_ind = self.sub2ind(row, col)

                        num_transitions_from_i = np.sum([
                            np.sum([
                                gamma_seqs[q][t][lin_ind]
                                for t in range(len(observations[0]) - 1)
                            ])
                            for q in range(len(observations))
                        ])

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

                                t_new[next_lin_ind, lin_ind] = num_transitions_from_i_to_j / num_transitions_from_i if num_transitions_from_i > 0 else 0

                        #         if t_new[next_lin_ind, lin_ind] > 1:
                        #             print(f"    {row,col} -> {next_row, next_col}: t {self.T[next_lin_ind, lin_ind]} -> {t_new[next_lin_ind, lin_ind]}")

                        total_transitions_from_i = np.sum([
                            t_new[next_lin_ind, lin_ind]
                            for next_lin_ind in range(self.numStates)
                        ])
                        for next_lin_ind in range(self.numStates):
                            t_new[next_lin_ind, lin_ind] = t_new[next_lin_ind, lin_ind] / total_transitions_from_i if total_transitions_from_i > 0 else 0

                self.T = t_new

                # Reset black room probabilities
                self.T[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
                self.T[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
                self.T[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
                self.T[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0

                # Check that all entries are between 0 and 1.
                if not np.all((np.min(self.T) >= 0) & (np.max(self.T) <= 1)):
                    raise ValueError("Transition matrix T contains entries outside the interval [0, 1].")

                # Check that each column sums to 1 (or 0 for cases with no probability mass).
                for i in range(self.numStates):
                    col_sum = np.sum(t_new[:, i])
                    if not (np.isclose(col_sum, 1.0, atol=tol)):
                        for next_row in range(self.envShape[0]):
                            for next_col in range(self.envShape[1]):
                                next_lin_ind = self.sub2ind(next_row, next_col)

                                num_transitions_from_i_to_j = np.sum([
                                    np.sum([
                                        xi_seqs[q][t][i, next_lin_ind]
                                        for t in range(len(observations[0]) - 1)
                                    ])
                                    for q in range(len(observations))
                                ])
                                num_transitions_from_i = np.sum([
                                    np.sum([
                                        gamma_seqs[q][t][i]
                                        for t in range(len(observations[0]) - 1)
                                    ])
                                    for q in range(len(observations))
                                ])
                                print(f"    At {next_lin_ind}, {i}: t {num_transitions_from_i_to_j, num_transitions_from_i}, {t_new[next_lin_ind, i]}")
                        raise ValueError(f"Column {i} in transition matrix T sums to {col_sum}, expected 1.")
                print("     Transition matrix T is valid.")

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
                                print(f"    {obs_ind} @ {row, col}: m {self.M[obs_ind, lin_ind]} -> {m_new[obs_ind, lin_ind], num_times_state_i_observe_m, num_times_state_i}")
                self.M = m_new

                # Reset black room probabilities ?
                self.M[:, self.sub2ind(0, 0)] = 0.25
                self.M[:, self.sub2ind(1, 1)] = 0.25
                self.M[:, self.sub2ind(0, 3)] = 0.25
                self.M[:, self.sub2ind(3, 2)] = 0.25

                # Check that all entries are between 0 and 1.
                if not np.all((np.min(self.M) >= 0) & (np.max(self.M) <= 1)):
                    raise ValueError("Emission matrix M contains entries outside the interval [0, 1].")

                # Loop through each column (each state) and check that the sum equals 1.
                num_states = self.M.shape[1]
                for i in range(num_states):
                    col_sum = np.sum(self.M[:, i])
                    if not np.isclose(col_sum, 1.0, atol=tol):
                        raise ValueError(f"Column {i} in emission matrix M sums to {col_sum}, expected 1.")

                print("     Emission matrix M is valid.")

                total_log_likelihood = -np.sum([
                    np.sum(np.log(c_seq))
                    for c_seq in c_seqs
                ])
                total_log_likelihoods.append(total_log_likelihood)

                print(f"Epoch {epoch}, Total log-likelihood: {total_log_likelihood}")
                # print(total_log_likelihoods)
                if abs(total_log_likelihoods[-1] - total_log_likelihoods[-2]) < epsilon:
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

            fileName = f"trained-model_epochs_{epoch}.pkl"
            print("Saving trained model as " + fileName)
            pickle.dump({'T': self.T, 'M': self.M, 'pi': self.pi}, open(fileName, "wb"))

    def viterbi2(self, observations):
        """Implement the Viterbi algorithm in log space."""
        delta_seq = []
        pre_seq = []

        # Convert priors to 1D array if not already
        self.pi = np.asarray(self.pi).flatten()

        for t, observation in enumerate(observations):
            obs_ind = self.obs2ind(observation)
            if t == 0:
                delta_seq.append([
                    np.log(self.M[obs_ind, i] + 1e-300) + np.log(self.pi[i] + 1e-300)
                    for i in range(self.numStates)
                ])
                pre_seq.append([None] * self.numStates)
                continue

            delta_t = []
            pre_t = []
            for curr_state in range(self.numStates):
                best_val = -np.inf
                best_prev = None
                for prev_state in range(self.numStates):
                    # Skip paths with zero probability by adding a small constant if needed.
                    val = np.log(self.T[curr_state, prev_state] + 1e-300) + delta_seq[t-1][prev_state]
                    if val > best_val:
                        best_val = val
                        best_prev = prev_state
                # Multiply by emission probability in log space (i.e., add its log)
                delta_val = np.log(self.M[obs_ind, curr_state] + 1e-300) + best_val
                delta_t.append(delta_val)
                pre_t.append(best_prev)
            delta_seq.append(delta_t)
            pre_seq.append(pre_t)

        # Backtracking to find the optimal state sequence
        x_seq = np.empty(len(observations), dtype=int)
        x_seq[-1] = int(np.argmax(delta_seq[-1]))

        for t in range(len(observations) - 2, -1, -1):
            x_seq[t] = pre_seq[t+1][x_seq[t+1]]

        # Convert linear indices to (i, j) coordinates.
        x_seq_new = [self.ind2sub(state) for state in x_seq]
        return x_seq_new

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

        # Forward Pass
        delta_seq = []
        pre_seq = []

        for t, observation in enumerate(observations):
            obs_ind = self.obs2ind(observation)

            if t == 0:
                delta_seq.append([
                    self.M[obs_ind, lin_index] * self.pi[lin_index]
                    for lin_index in range(self.numStates)
                ])
                pre_seq.append([None] * self.numStates)
                continue

            delta_t = []
            pre_t = []

            for lin_index in range(self.numStates):
                best_val = -np.inf
                best_prev = -1
                for prev_lin_index in range(self.numStates):
                    val = self.T[lin_index, prev_lin_index] * delta_seq[t-1][prev_lin_index]
                    if val > best_val:
                        best_val = val
                        best_prev = prev_lin_index

                delta_val = self.M[obs_ind, lin_index] * best_val
                delta_t.append(delta_val)
                pre_t.append(best_prev)

            delta_seq.append(delta_t)
            pre_seq.append(pre_t)

        # Select most likely terminal state
        x_seq = np.empty(len(observations))
        x_seq[:] = np.nan
        x_seq[len(observations) - 1] = np.argmax(delta_seq[-1])
        print(f"    Determined last as {x_seq[-1], delta_seq}")

        # Backward Pass
        for t in range(len(observations) - 2, -1, -1):
            x_seq[t] = pre_seq[t+1][int(x_seq[t+1])]
            print(f"        {t}, Pointed from {x_seq[t+1]} to {x_seq[t]}")

        x_seq_new = [
            self.ind2sub(int(x))
            for x in x_seq
        ]

        # print(x_seq_new)

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
