# ----- ECG classification by Maxat Sangerbaev (16BD02042) and Markitanov Denis (16BD02042)
# For Machine Learning course (Fall 2018) KBTU

import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from customMiniBatch import MiniBatchKMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Accuracy check
def acc(real, predicted):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(real)):
        if real[i] == predicted[i] and real[i] == 1: TP += 1
        elif real[i] == predicted[i] and real[i] == 0: TN += 1
        elif real[i] != predicted[i] and real[i] == 1: FP += 1
        else: FN += 1
    return ((TP + TN) / (TP + TN + FP + FN)) * 100

# Average function for all the NaN values in dataset
def avg(array):
    avg_holder = []
    for i in range(9):
        holder = 0
        for j in range(int(len(array[0]) / 10)):
            holder += array[i][j]
        avg_holder.append(int(holder / (int(len(array[0]) / 10))))
    return avg_holder

# Read data from the Electrocardiogram.csv file
# data_set_raw = []
# with open("Electrocardiogram.csv", "r") as data:
#     data = csv.reader(data, delimiter = ",")
#     data_read = [row for row in data]

# # Data preparation
# data_read.pop(0)
# for row in data_read:
#     a = row[0].split(",")
#     for i in range(4):
#         a.pop(0)
#     for i in range(len(a)):
#         if a[i] != '""':
#             a[i] = a[i][1:-1]
#             a[i] = int(a[i])
#         else:
#             a[i] = 0
#     data_set_raw.insert(0, a)
# avg = avg(data_set.T)
# for i in range(len(data_set)):
#     for j in range(len(data_set[0] - 1)):
#         if data_set[i][j] == 0: data_set[i][j] = avg[j]

# Data serialization
# binary_file = open('data_set.bin',mode='wb')
# data_set_bin = pickle.dump(data_set, binary_file)
# binary_file.close()

# ----- Main ECG

# Deserialization of formatted data
data_set_raw = pickle.load( open( "data_set.bin", "rb" ) ) # we take out serialized dataset (raw)
data_set = np.empty([9, 1000000])
data_set = np.array(data_set_raw) # present raw dataset as numpy.array

# Data preprocessing
data_set = np.delete(data_set, np.s_[3:9], axis=1) # take out not important features
train_data = data_set[:950000] # training data
test_data = data_set[-10:] # preusdo random test

# Transformation of data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)

# KMEANS clusterization
model = MiniBatchKMeans(n_clusters=2, init="k-means++", batch_size=100) # create instance of MiniBatchKMeans with 2 clusters
all_predictions_train = model.fit_predict(train_data)
all_predictions_test = model.predict(test_data)
print(all_predictions_test)
real = [1, 1, 1, 1, 0, 1, 0, 0, 1, 0]
real1 = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1]
print(acc(real, all_predictions_test))
print(acc(real1, all_predictions_test))

# Visualization
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_title("train_data")
ax.set_xlabel("RR")
ax.set_ylabel("PR")
ax.set_zlabel("QRS")
ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], c=all_predictions_train, s=50, cmap='viridis')
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title("test_data")
ax.set_xlabel("RR")
ax.set_ylabel("PR")
ax.set_zlabel("QRS")
ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=all_predictions_test, s=50, cmap='viridis')
plt.show(block=True)