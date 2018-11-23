import numpy as np
import csv
import pandas as pd
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

def avg(array):
    avg_holder = []
    for i in range(9):
        holder = 0
        for j in range(int(len(array[0]) / 10)):
            holder += array[i][j]
        avg_holder.append(int(holder / (int(len(array[0]) / 10))))
    return avg_holder

# # class Advice(Enum):
# #     NO = 0
# #     YES = 1
# #     P = 2
# data_set_raw = []
# открыл файл, считал все данные
# with open("Electrocardiogram.csv", "r") as data:
#     data = csv.reader(data, delimiter = ",")
#     data_read = [row for row in data]

# # Data preparation
# Убрал первую строку (есть комменты на английском иногда, я из до этого оставлял)
# data_read.pop(0)

# Цикл ниже делает несколько вещей: избавляется от запятых в файле, убирает первые 4 столбца, переводит все в инт, записывает в главнцю матрицу (data_set)
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
# Кстати, цикл выше еще заменяет пустые значения нулями
# А тут (ниже), я заменяю нули на среднее (функция среднегно выше)
# avg = avg(data_set.T)
# for i in range(len(data_set)):
#     for j in range(len(data_set[0] - 1)):
#         if data_set[i][j] == 0: data_set[i][j] = avg[j]
# Тут (ниже) я записал датасет в файл, что бы считывать его быстрее
# binary_file = open('data_set.bin',mode='wb')
# data_set_bin = pickle.dump(data_set, binary_file)
# binary_file.close()
# Тут (ниже) я просто вскрываю файл и записываю все в главную матрицу (data_set)
data_set_raw = pickle.load( open( "data_set.bin", "rb" ) )
data_set = np.empty([9, 1000000])
data_set = np.array(data_set_raw)
kf = KFold(n_splits=2)
for train_index, test_index in kf.split(data_set):
    X_train, X_test = data_set[train_index], data_set[test_index]
for train_index, test_index in kf.split(X_train):
    X1_train, X1_test = X_train[train_index], X_train[test_index]
for train_index, test_index in kf.split(X1_train):
    X2_train, X2_test = X1_train[train_index], X1_train[test_index]
for train_index, test_index in kf.split(X2_train):
    X3_train, X3_test = X2_train[train_index], X2_train[test_index]
# Нижние строки используют для того, что бы минимизировать данные (они в коммента ибо плохо работают при кластеризации)
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(data_set)
# data_set = scaler.transform(data_set)

# KMEANS clusterization
model = MiniBatchKMeans(n_clusters=2, init="k-means++", n_init=1000)
model.fit(X3_train)
all_predictions = model.predict(X3_test)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
plt.scatter(X3_test[:, 0], X3_test[:, 1], c=all_predictions, s=50, cmap='viridis')
plt.show(block=True)

# centers = model.cluster_centers_
# plt.scatter(data_set[:, 0], data_set[:, 1], c='black', s=200, alpha=0.5)
# plt.show(block = True)