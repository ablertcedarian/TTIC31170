#%%
import sys
import pickle
from HMM import HMM
from HMM2 import HMM2
from DataSet import DataSet

# This function should be called with one argument: trainingdata.txt

dataset = DataSet("randomwalk.train.txt")
dataset.readFile()

#%%
baseline = HMM(dataset.envShape)
#%%
from HMM_reborn import HMM_reborn
hmm = HMM_reborn(dataset.envShape)
hmm.train(dataset.observations, dataset.states, True)

#%%
baseline.pi, hmm.pi

#%%
baseline.T, hmm.T

#%%
baseline.M, hmm.M

#%%
import matplotlib.pyplot as plt

#%%
likelihoods = [
    53947.6623176750,
-278671.4060098913,
-288918.9545975897,
-304693.2441273201,
-318249.9088999329,
-327989.3177353394,
-333273.8968124922,
-344958.3555107749,
-365025.6676066880,
-370382.2393821460,
-370553.1572437847,
-370553.4008327170,
-370553.4008333705,
-370553.4008333705,
]

plt.figure(figsize=(8, 5))
plt.plot(likelihoods, marker='o')
plt.title("Log-Likelihood vs Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("Total Log-Likelihood")
plt.grid(True)
plt.tight_layout()
plt.savefig("log_likelihood_curve.png")
plt.show()

#%%
# Save the model for future use
fileName = "trained-model_REBORN_1.pkl"
print("Saving trained model as " + fileName)
pickle.dump({'T': hmm.T, 'M': hmm.M, 'pi': hmm.pi}, open(fileName, "wb"))
