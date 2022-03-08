# import required python libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from hydroeval import evaluator, nse, kge
import itertools

class Machine_Learning():

    def __init__(self, input, separator):
        self.input = input
        self.separator = separator

    def preprocessing(self):
        self.total_input = pd.read_csv(self.input)
        input = self.total_input.columns[self.separator[1]:self.separator[2]]
        label = self.total_input.columns[self.separator[0]:self.separator[1]]
        input_sentinel = self.total_input[input]
        label_algae = self.total_input[label]
        return input_sentinel, label_algae

# RandomForest Regression Algorithm
def random_forest(X_train, Y_train, X_test, Y_test, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, np.ravel(Y_train, order="C"))
    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    return [model, X_train, Y_train, X_test, Y_test, Y_train_predict, Y_test_predict]
    
# GBR(GradientBoostingRegression) Algorithm
def gradient_boosting(X_train, Y_train, X_test, Y_test, n_estimators=100, max_depth=3):
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, np.ravel(Y_train, order="C"))
    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    return [model, X_train, Y_train, X_test, Y_test, Y_train_predict, Y_test_predict]

# XGBoosting Algorithm
def xgboosting(X_train, Y_train, X_test, Y_test, n_estimators=100, learning_rate=0.08, max_depth=3):
    model = xgboost.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train, np.ravel(Y_train, order="C"))
    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    return [model, X_train, Y_train, X_test, Y_test, Y_train_predict, Y_test_predict]


def algae_monitor(input_sentinel, label_algae, input_col = [], model_list=["RF"], trainSize_rate=0.8, n_estimators=100, random_state=42, max_depth=3, learning_rate=0.08):

    combination_list = []
    if not input_col:
        input_col_list = list(input_sentinel)
        for i in range(1, len(input_col_list) + 1):
            combination_list.append(list(map(' '.join, itertools.combinations(input_col_list, i))))
    else:
        combination_list = input_col

    parameters_list= []
    for i in combination_list:
        parameters_list.extend(i)
    #val_data_col = list(label_algae)

    #performance_index_list = []
    results = []
    for p in combination_list:
        for p_value in p:
            ## 
            x_data = input_sentinel.reindex(columns=p_value.split(' '))
            print("parameter :", p_value)

            for md in model_list:
                count = 0

                for l in label_algae:
                    """
                    Modulation 2 : 학습데이터 정제
                    """
                    # 
                    combined_df = pd.concat([x_data, label_algae[l]], axis=1)

                    dataDim = len(combined_df.columns)  # 매개변수의 개수

                    # 
                    trainSize = int(len(combined_df) * trainSize_rate)
                    trainSet = combined_df[0:trainSize]
                    testSet = combined_df[trainSize:]

                    X_train = trainSet.drop([trainSet.columns[-1]], axis='columns')
                    Y_train = trainSet.iloc[:, -1]

                    X_test = testSet.drop([testSet.columns[-1]], axis='columns')
                    Y_test = testSet.iloc[:, -1]
                    """
                    Modulation 3 : 모델 학습
                    """
                    # 
                    if md == "RF":

                        # 각 지점의 알맞는 학습 배치 데이터 생성
                        result = random_forest(X_train, Y_train, X_test, Y_test, n_estimators, random_state)

                    elif md == "GBR":
                        result = gradient_boosting(X_train, Y_train, X_test, Y_test, n_estimators, max_depth)

                    elif md == "XGB":
                        result = xgboosting(X_train, Y_train, X_test, Y_test, n_estimators, learning_rate, max_depth)

                    else:
                        print("Please change the method. There are options: 'RF', 'GBR', and 'XGB' in the current version")
        
                    results.append(result)

    return results

# Performance Test
def performance_test(pt, resutls):

    if pt == "R2":
        score_train = r2_score(resutls[2], resutls[5])
        score_test = r2_score(resutls[4], resutls[6])
    if pt == "MSE":
        score_train = mean_squared_error(resutls[2], resutls[5])
        score_test = mean_squared_error(resutls[4], resutls[6])
    if pt == "MAE":
        score_train = mean_absolute_error(resutls[2], resutls[5])
        score_test = mean_absolute_error(resutls[4], resutls[6])
    if pt == "RMSE":
        score_train = mean_squared_error(resutls[2], resutls[5])**0.5
        score_test = mean_squared_error(resutls[4], resutls[6])**0.5
    if pt == "NSE":
        score_train = evaluator(nse, resutls[2], resutls[5], axis=1)
        score_test = evaluator(nse, resutls[4], resutls[6], axis=1)
    if pt == "KGE":
        score_train = evaluator(kge, resutls[2], resutls[5], axis=1)
        score_test = evaluator(kge, resutls[4], resutls[6], axis=1)

    return score_train, score_test