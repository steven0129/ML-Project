import sys
import feature_selection
from model import lstm
import numpy as np

train_path = sys.argv[1]
test_path = sys.argv[2]
# path = r'full1_upload/training_data/Training freq 1D, OW 1, PW 1.csv'

def data_process(path):
    if(path[14] == 'r'):
        bias = 3  # 訓練集需偏移 3 格找到資料
    else:
        bias = 4  # 測試集需偏移 4 格找到資料
    fp = open(path, 'r')
    for i in range(4):
        line = fp.readline()  # 跳過標題欄位

    data = []
    label = []
    while line:
        tmp = []
        no_label = False
        for i in range(1, 85):
            idx = line.find(',')  # 尋找 "," 的位置
            if(i < bias):
                line = line[idx+1:]  # 小於偏移值代表尚未找到資料 (還在標題欄)，直接跳到下一個 "," 之後
                continue
            if(bias == 3 and i == 84):  # 這一格是 label
                if(line[idx+1:-1] == 'True'):
                    label.append([1.0, 0.0])  # 人工 One-hot，是不是很直觀
                elif(line[idx+1:-1] == 'False'):
                    label.append([0.0, 1.0])
                else:
                    no_label = True  # 沒有 label
                break  # 找到或是沒找到 label ，沒必要繼續再找資料，直接跳行
            value = line[:idx]
            if(value == ''):
                value = 0  # 該欄位沒資料，放 0
            tmp.append(float(value))
            line = line[idx+1:]
        line = fp.readline()  # 跳行
        if(bias == 4 or not no_label):  # 該筆資料集如果是測試資料或是一筆有 label 的訓練資料
            data.append(tmp)  # 放入該筆資料的所有特徵值
        del tmp
    fp.close()
    return data, label

data, label = data_process(train_path)

train_point = [0, 989, 1964, 2947, 3930, 4957, 5939, 6854, 7839, 8814, 9725, 10667, 11551]
x_train = []
for i in range(len(train_point) - 1):
    x_train.append(data[train_point[i]:train_point[i + 1]])

y_train = []
for i in range(len(train_point) - 1):
    y_train.append(label[train_point[i]:train_point[i + 1]])

data, label = data_process(test_path)

test_point = [0, 960, 1899, 2898, 3283]
x_test = []
for i in range(len(test_point) - 1):
    x_test.append(data[test_point[i]:test_point[i + 1]])

for i in range(len(x_train)):
    x_train[i].extend([[0.0] * 81] * (1027 - len(x_train[i])))
    y_train[i].extend([[0.0, 1.0]] * (1027 - len(y_train[i])))

for i in range(len(x_test)):
    x_test[i].extend([[0.0] * 81] * (1027 - len(x_test[i])))

print(np.array(x_train).shape)
print(np.array(x_test).shape)
print(np.array(y_train).shape)

Model = lstm()
Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Model.fit(np.array(x_train), np.array(y_train), epochs=100, batch_size=64)
predictions = Model.predict(np.array(x_test))

results = []
lengths = [960, 939, 999, 385]
for idx, pred in enumerate(predictions.tolist()):
    results.extend(pred[0:lengths[idx]])

print(np.array(results))
np.savetxt('result/result.csv', np.array(results), delimiter=',')

# Feature Selection
'''
importances = feature_selection.RF(data, label)
for index, value in importances:
    print(f'feature {index}: {value}')
'''
