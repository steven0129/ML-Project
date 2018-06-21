import numpy as np
from sklearn.ensemble import RandomForestClassifier

def RF(data, label): # 使用Random Forest進行特徵選取
    # 轉為numpy array
    X = np.array(data)
    y = np.array(label)
    
    # 訓練RandomForest Classifier
    RF = RandomForestClassifier(max_depth=2, random_state=0)
    RF.fit(X, y)
    
    # 存放每個特徵的重要性
    importances = RF.feature_importances_
    
    # 排序重要性
    indices = np.argsort(importances)[::-1] # [::-1]代表反轉陣列
    importances = importances[indices]
    
    # zip([1, 2, 3], [4, 5, 6]) == [(1, 4), (2, 5), (3, 6)]
    return list(zip(indices, importances))