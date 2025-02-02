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

x_train = []
x_train.append(data)
y_train = []
y_train.append(label)

data, label = data_process(test_path)

x_test = []
x_test.append(data)
#y_test = []
#y_test.append(label)

for i in range(len(x_test[0]), len(x_train[0])):
    x_test[0].append([0.0] * 81)

print(x_test)
print(np.array(x_train).shape)
print(np.array(x_test).shape)

Model = lstm()
Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Model.fit(np.array(x_train), np.array(y_train), epochs=40, batch_size=64)
result = Model.predict(np.array(x_test))
print(result)
np.savetxt('result/result.csv', result[0], delimiter=',')

# Feature Selection
'''
importances = feature_selection.RF(data, label)
for index, value in importances:
    print(f'feature {index}: {value}')
'''
