#%%
import sys
import pickle

from HMM import HMM
from DataSet import DataSet

f = open("trained-model.pkl", "rb")
model = pickle.load(f)
T = model['T']
M = model['M']
pi = model['pi']

dataset = DataSet("randomwalk.test.txt")
dataset.readFile()

hmm = HMM(dataset.envShape, T, M, pi)

#%%
pi
#%%
T

#%%
f = open("trained-model_1.pkl", "rb")
model = pickle.load(f)
T_2 = model['T']
M_2 = model['M']
pi_2 = model['pi']

#%%
pi_2
#%%
f = open("trained-model_epochs_2.pkl", "rb")
model = pickle.load(f)
T_3 = model['T']
M_3 = model['M']
pi_3 = model['pi']

#%%
pi_3

#%%
f = open("trained-model_epochs_5.pkl", "rb")
model = pickle.load(f)
T_5 = model['T']
M_5 = model['M']
pi_5 = model['pi']

#%%
pi_5
#%%
T_5
#%%
T_3

#%%
f = open("trained-model_epochs_0.pkl", "rb")
model = pickle.load(f)
T_5 = model['T']
T_5
#%%

import numpy as np

dataset = DataSet("randomwalk.train.txt")
dataset.readFile()

# hmm = HMM(dataset.envShape)
# hmm.train(dataset.observations, dataset.states)

# Helper functions
def sub2ind(i, j, ncols):
    """Convert 2D grid coordinates to a 1D index (row-major order)."""
    return i * ncols + j

def obs2ind(obs):
    """Map observation symbol to index.
       In our emission matrix, rows correspond to:
       'r' --> 0, 'g' --> 1, 'b' --> 2, 'y' --> 3.
    """
    mapping = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
    return mapping[obs]

# Define HMM parameters based on a 4x4 grid.
envShape = [4, 4]
nrows, ncols = envShape
numStates = nrows * ncols

# -----------------------------------
# 1. Build the Transition Matrix T.
# T is a numStates x numStates array where:
# T[i, j] is the probability of transitioning from state j (previous state) to state i (current state).
T = np.zeros((numStates, numStates))

# Default self-transitions: each state has 0.2 chance to stay
for i in range(numStates):
    T[i, i] = 0.2

# Black room states: (0,0), (1,1), (0,3), (3,2)
black_coords = [(0, 0), (1, 1), (0, 3), (3, 2)]
for (i, j) in black_coords:
    idx = sub2ind(i, j, ncols)
    T[idx, idx] = 1.0

# Add additional transitions from the snippet:
# (1,0) --> (2,0)
T[sub2ind(2, 0, ncols), sub2ind(1, 0, ncols)] = 0.8

# (2,0) transitions:
T[sub2ind(1, 0, ncols), sub2ind(2, 0, ncols)] = 0.8 / 3.0
T[sub2ind(2, 1, ncols), sub2ind(2, 0, ncols)] = 0.8 / 3.0
T[sub2ind(3, 0, ncols), sub2ind(2, 0, ncols)] = 0.8 / 3.0

# (3,0) transitions:
T[sub2ind(2, 0, ncols), sub2ind(3, 0, ncols)] = 0.8 / 2.0
T[sub2ind(3, 1, ncols), sub2ind(3, 0, ncols)] = 0.8 / 2.0

# (0,1) --> (0,2)
T[sub2ind(0, 2, ncols), sub2ind(0, 1, ncols)] = 0.8

# (2,1) transitions:
T[sub2ind(2, 0, ncols), sub2ind(2, 1, ncols)] = 0.8 / 3.0
T[sub2ind(3, 1, ncols), sub2ind(2, 1, ncols)] = 0.8 / 3.0
T[sub2ind(2, 2, ncols), sub2ind(2, 1, ncols)] = 0.8 / 3.0

# (3,1) transitions:
T[sub2ind(2, 1, ncols), sub2ind(3, 1, ncols)] = 0.8 / 2.0
T[sub2ind(3, 0, ncols), sub2ind(3, 1, ncols)] = 0.8 / 2.0

# (0,2) transitions:
T[sub2ind(0, 1, ncols), sub2ind(0, 2, ncols)] = 0.8 / 2.0
T[sub2ind(1, 2, ncols), sub2ind(0, 2, ncols)] = 0.8 / 2.0

# (1,2) transitions:
T[sub2ind(0, 2, ncols), sub2ind(1, 2, ncols)] = 0.8 / 3.0
T[sub2ind(2, 2, ncols), sub2ind(1, 2, ncols)] = 0.8 / 3.0
T[sub2ind(1, 3, ncols), sub2ind(1, 2, ncols)] = 0.8 / 3.0

# (2,2) transitions:
T[sub2ind(1, 2, ncols), sub2ind(2, 2, ncols)] = 0.8 / 3.0
T[sub2ind(2, 1, ncols), sub2ind(2, 2, ncols)] = 0.8 / 3.0
T[sub2ind(2, 3, ncols), sub2ind(2, 2, ncols)] = 0.8 / 3.0

# (1,3) transitions:
T[sub2ind(1, 2, ncols), sub2ind(1, 3, ncols)] = 0.8 / 2.0
T[sub2ind(2, 3, ncols), sub2ind(1, 3, ncols)] = 0.8 / 2.0

# (2,3) transitions:
T[sub2ind(1, 3, ncols), sub2ind(2, 3, ncols)] = 0.8 / 3.0
T[sub2ind(3, 3, ncols), sub2ind(2, 3, ncols)] = 0.8 / 3.0
T[sub2ind(2, 2, ncols), sub2ind(2, 3, ncols)] = 0.8 / 3.0

# (3,3) --> (2,3)
T[sub2ind(2, 3, ncols), sub2ind(3, 3, ncols)] = 0.8

# -----------------------------------
# 2. Build the Emission Matrix M.
# M is a 4 x numStates array where M[k, i] is the probability of observing symbol k from state i.
# Initialize with default 0.1 and then override for special cases.
M = np.ones((4, numStates)) * 0.1

# For black states, set all emissions to 0.25 (they will not be used since pi for these is zero).
for (i, j) in black_coords:
    idx = sub2ind(i, j, ncols)
    M[:, idx] = 0.25

# Set boosted emission probabilities for specific states:
# (0,1): boosted for 'r'
M[obs2ind('r'), sub2ind(0, 1, ncols)] = 0.7
# (0,2): boosted for 'g'
M[obs2ind('g'), sub2ind(0, 2, ncols)] = 0.7
# (1,0): boosted for 'g'
M[obs2ind('g'), sub2ind(1, 0, ncols)] = 0.7
# (1,2): boosted for 'b'
M[obs2ind('b'), sub2ind(1, 2, ncols)] = 0.7
# (1,3): boosted for 'r'
M[obs2ind('r'), sub2ind(1, 3, ncols)] = 0.7
# (2,0): boosted for 'y'
M[obs2ind('y'), sub2ind(2, 0, ncols)] = 0.7
# (2,1): boosted for 'g'
M[obs2ind('g'), sub2ind(2, 1, ncols)] = 0.7
# (2,2): boosted for 'r'
M[obs2ind('r'), sub2ind(2, 2, ncols)] = 0.7
# (2,3): boosted for 'y'
M[obs2ind('y'), sub2ind(2, 3, ncols)] = 0.7
# (3,0): boosted for 'b'
M[obs2ind('b'), sub2ind(3, 0, ncols)] = 0.7
# (3,1): boosted for 'y'
M[obs2ind('y'), sub2ind(3, 1, ncols)] = 0.7
# (3,3): boosted for 'b'
M[obs2ind('b'), sub2ind(3, 3, ncols)] = 0.7

# -----------------------------------
# 3. Define the Prior Distribution pi.
# Non-black states get 1/12 probability and black states get 0.
pi = np.ones(numStates) / 12.0
for (i, j) in black_coords:
    idx = sub2ind(i, j, ncols)
    pi[idx] = 0.0

# -----------------------------------
# 4. Create an Observation Sequence of 200 time steps.
# For demonstration, we generate a random sequence of observations.
np.random.seed(0)  # For reproducibility.
# observations = np.random.choice(['r', 'g', 'b', 'y'], size=200)
observations = dataset.observations[0]

# -----------------------------------
# 5. Forward Algorithm to Calculate Normalized Alphas.
def forward_algorithm(obs_seq, T, M, pi):
    """
    Compute the normalized forward messages for a given observation sequence.

    Parameters:
      obs_seq: array-like sequence of observation symbols.
      T: Transition matrix of shape (numStates, numStates), where T[i, j] is the probability
         of transitioning from state j to state i.
      M: Emission matrix of shape (numObsSymbols, numStates), where M[k, i] is the probability
         of emitting observation with index k in state i.
      pi: Prior distribution over states (1D array of length numStates).

    Returns:
      alphas: A 2D array of shape (len(obs_seq), numStates) with normalized forward messages.
    """
    T_steps = len(obs_seq)
    numStates = len(pi)
    alphas = np.zeros((T_steps, numStates))

    # t = 0 (initialization)
    first_obs = obs_seq[0]
    o_idx = obs2ind(first_obs)
    alphas[0, :] = pi * M[o_idx, :]
    c = np.sum(alphas[0, :])
    if c != 0:
        alphas[0, :] /= c

    # Recursion for t >= 1
    for t in range(1, T_steps):
        o_idx = obs2ind(obs_seq[t])
        for i in range(numStates):
            # Sum over previous states j: note that T[i, j] is P(x_t = i | x_(t-1) = j)
            alphas[t, i] = M[o_idx, i] * np.sum(T[i, :] * alphas[t - 1, :])
        c = np.sum(alphas[t, :])
        if c != 0:
            alphas[t, :] /= c  # Normalize
    return alphas

# Compute the normalized alphas for the 200-step observation sequence.
normalized_alphas = forward_algorithm(observations, T, M, pi)

# -----------------------------------
# 6. Print some results.
print("First 5 normalized alpha vectors:")
print(normalized_alphas[:5])
print("\nLast normalized alpha vector (time step 200):")
print(normalized_alphas[-1])
