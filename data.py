import sys
import feature_selection

path = sys.argv[1]
# path = r'full1_upload/training_data/Training freq 1D, OW 1, PW 1.csv'

if(path[14] == 'r'):
    bias = 3  # 訓練集需偏移 3 格找到資料
else:
    bias = 4  # 測試集需偏移 4 格找到資料

fp = open(path, 'r')
for i in range(4):
    line = fp.readline()  # 跳過標題欄位

data = []
label= []
while line:
    tmp = []
    no_label = False
    for i in range(1, 85):
        idx = line.find(',')  # 尋找 "," 的位置
        if(i < bias):
            line = line[idx+1:]  # 小於偏移值代表尚未找到資料 (還在標題欄)，直接跳到下一個 "," 之後
            continue
        if(i == 84):  # 這一格是 label
            if(line[idx+1:-1] == 'True'):
                label.append([1, 0])  # 人工 One-hot，是不是很直觀
            elif(line[idx+1:-1] == 'False'):
                label.append([0, 1])
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

# Test
print(len(label))
print(len(data))

# Feature Selection
indices, importances = feature_selection.RF(data, label)
for f in range(len(data[0])):
    print(f'{f + 1}. feature {indices[f]} ({importances[indices[f]]})')