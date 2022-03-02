import matplotlib.pyplot as plt
import numpy as np

# 입출력 자료 통합차트
def linear_regression(X, Y, method, score_test):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(X, Y, 'o')
    m, b = np.polyfit(X, Y, 1)
    plt.xlim([0, 70])
    plt.ylim([0, 70])
    #add linear regression line to scatterplot 
    plt.plot(X, m*X+b, label='Prediction = {:.2f} X Observation + {:.2f}, {:.5s}={:.2f}'.format(m,b, method, score_test))
    plt.legend(fontsize=12)
    plt.show()