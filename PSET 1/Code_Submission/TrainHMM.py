import sys
import pickle
from HMM import HMM
from HMM2 import HMM2
from HMM_reborn import HMM_reborn
from DataSet import DataSet

if __name__ == '__main__':
    """Read in data, call code to train HMM, and save model."""

    # This function should be called with one argument: trainingdata.txt
    if (len(sys.argv) != 2):
        print("Usage: TrainMM.py trainingdata.txt")
        sys.exit(0)

    dataset = DataSet(sys.argv[1])
    dataset.readFile()

    # hmm = HMM(dataset.envShape)
    hmm = HMM_reborn(dataset.envShape)
    hmm.train(dataset.observations, dataset.states)

    # Save the model for future use
    fileName = "trained-model_REBORN_1.pkl"
    print("Saving trained model as " + fileName)
    pickle.dump({'T': hmm.T, 'M': hmm.M, 'pi': hmm.pi}, open(fileName, "wb"))
