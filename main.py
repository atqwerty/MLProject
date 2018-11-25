import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from customMiniBatch import MiniBatchKMeans

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
train_data = data_set[:90000]
test_data = data_set[100000:100010]

# KMEANS clusterization
model = MiniBatchKMeans(n_clusters=2, init="k-means++", n_init=100)
all_predictions = model.fit_predict(train_data)
# all_predictions = model.predict(test_data)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
plt.scatter(train_data[:, 0], train_data[:, 1], c=all_predictions, s=50, cmap='viridis')
plt.show(block=True)