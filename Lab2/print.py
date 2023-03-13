import numpy as np
import matplotlib.pyplot as plt


accuracies = np.array([0.901, 0.901, 0.585, 0.73, 0.9])
network_sizes = np.array([28.76, 14.49, 1, 1, 8.177863])
legends = np.array(["FP32", "FP16", "Binary Connect 1", "Binary Connect 2", "Int8"])

accuracies2 = np.array([0.901, 0.885])
legends2 = np.array(["1", "2"])

plt.figure()
plt.bar(legends2, accuracies2)
plt.grid()
plt.ylim(0.5, 1.0)
plt.ylabel("Validation Accuracy")
plt.title("Analysis of different architectures")
plt.savefig("Lab2/test3.png")
plt.show()

plt.figure()
plt.bar(legends, accuracies)
plt.grid()
plt.ylabel("Validation Accuracy")
plt.title("Analysis of different architectures")
plt.savefig("Lab2/test1.png")
plt.show()

plt.figure()
plt.bar(legends, network_sizes)
plt.grid()
plt.ylabel("Network Size (in MB)")
plt.title("Analysis of different architectures")
plt.savefig("Lab2/test2.png")
plt.show()

plt.figure()
plt.scatter(network_sizes, accuracies)
for i, txt in enumerate(legends):
    plt.annotate(txt, (network_sizes[i], accuracies[i]))
plt.grid()
plt.xlabel("Network size(in MB)")
plt.ylabel("Validation Accuracy")
plt.title("Analysis of different architectures")
plt.savefig("Lab2/test.png")
plt.show()