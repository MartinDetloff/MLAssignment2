from q2 import make_classification
from sklearn.svm import LinearSVC
import time
import itertools
import pandas as pd
from sklearn import datasets
import numpy as np
pd.options.display.float_format = '{:,.4f}'.format
dimensions = [10, 50, 100, 500, 1000]
num_samples = [500, 1000, 5000, 10000, 100000]
tests = list(itertools.product(dimensions, num_samples))
training_times = pd.DataFrame(tests, columns=['dimensions', 'num_samples'])

print(training_times.head())


random_seed = 1
DualAccuracy = []
DualTimeCost = []
PrimalAccuracy = []
PrimalTimeCost = []

for dimension in dimensions:
    for num_sample in num_samples:
        X_train, X_test, y_train, y_test = make_classification(dimension, num_sample, 1.0, random_seed)

        primalModel = LinearSVC(dual=False, random_state=1, max_iter=100000)

        start_time = time.time()
        primalModel.fit(X_train, y_train)
        end_time = time.time()

        training_time = end_time - start_time
        training_accuracy = primalModel.score(X_test, y_test)
        PrimalAccuracy.append(training_accuracy)
        PrimalTimeCost.append(training_time)

        #training_times[(dimension, num_sample)] = training_time
        print(f"PRIMAL MODEL: Training Accuracy: {training_accuracy : .4f} Training time for d={dimension}, n={num_sample}: {training_time:.4f} seconds")

        dualModel = LinearSVC(dual=True, random_state=1, max_iter=100000)

        start_time = time.time()
        dualModel.fit(X_train, y_train)
        end_time = time.time()

        training_time = end_time - start_time
        training_accuracy = dualModel.score(X_test, y_test)
        DualAccuracy.append(training_accuracy)
        DualTimeCost.append(training_time)

        #training_times[(dimension, num_sample)] = training_time
        print(f"DUAL MODEL: Training Accuracy: {training_accuracy : .4f} Training time for d={dimension}, n={num_sample}: {training_time:.4f} seconds")

training_times["PrimalAccuracy"] = PrimalAccuracy
training_times["PrimalTimeCost"] = PrimalTimeCost
training_times["DualAccuracy"] = DualAccuracy
training_times["DualTimeCost"] = DualTimeCost

print(training_times.head())

training_times.to_csv("training_times.csv",index=False,float_format='%.4f')
