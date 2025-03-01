import time

from matplotlib import pyplot as plt

from q2 import make_classification
from LinearSVC import LinearSVC

dimensions = [10, 50, 100, 500, 1000]
num_samples = [500, 1000, 5000, 10000, 100000]

training_times = {}

random_seed = 1

for d in dimensions:
    for n in num_samples:
        X_train, X_test, y_train, y_test = make_classification(d, n, 1.0, random_seed)
        svc = LinearSVC()
        start_time = time.process_time()
        svc.fit(X_train, y_train)
        end_time = time.process_time()
        training_time = end_time - start_time
        training_times[(d, n)] = training_time
        print(f"Training time for d={d}, n={n}: {training_time:.4f} seconds")

for d in dimensions:
    times = [training_times[(d, n)] for n in num_samples]
    plt.plot(num_samples, times, label=f"d={d}")

plt.xlabel("Number of Samples")
plt.ylabel("Training Time")
plt.title("Training Time vs. Number of Samples")
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()

for n in num_samples:
    times = [training_times[(d, n)] for d in dimensions]
    plt.plot(dimensions, times, label=f"n={n}")

plt.xlabel("Dimensions")
plt.ylabel("Training Time")
plt.title("Training Time vs. Dimensions")
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()