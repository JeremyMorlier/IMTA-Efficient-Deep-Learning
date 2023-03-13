import numpy as np
import matplotlib.pyplot as plt


accuracies = np.array([0.855, 0.882, 0.883, 0.862])
trainingParameters = np.array([4572300, 4038372, 7048548, 12643172])

plt.scatter(trainingParameters, accuracies)
plt.plot(trainingParameters, accuracies, color="blue", alpha=0.4)
plt.grid()
plt.xlabel("Number of training parameters")
plt.ylabel("Validation Accuracy")
plt.title("Analysis of different architectures")
plt.savefig("test.png")