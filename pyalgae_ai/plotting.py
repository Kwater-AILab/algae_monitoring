import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', family='HCR Dotum')

# 입출력 자료 통합차트
def total_chart(df1, list1):
    columns = list(range(1, len(list1)))
    i = 1
    values = df1.values
    plt.figure(figsize=(9, 40))
    for variable in columns:
        plt.subplot(len(columns), 1, i)
        plt.plot(values[:, variable])
        plt.title(df1.columns[variable], y=0.5, loc='right')
        i += 1
    plt.show()


#  기본 차트 : 이미지 저장용
def basic_chart(obsY, preY, str_part):
    if str_part == 'line':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(preY)), preY, '-x', label="predict Y")
        ax.plot(range(len(obsY)), obsY, '-', label="Original Y")
    plt.legend(loc='upper right')


#  R^2 차트 : 이미지 저장용
def r_square_chart(obsY, preY, str_part):
    obsY = obsY.values
    # print(obsY)
    # print(preY)
    # exit(1)
    if str_part == 'line':
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(np.unique(obsY), np.poly1d(np.polyfit(obsY, preY, 1))(np.unique(obsY)), color='#000000')

        ax.scatter(obsY, preY)

        # ax.plot(range(len(preY)), preY, '-x', label="predict Y")
        # ax.plot(range(len(obsY)), obsY, '-', label="Original Y")
    plt.legend(loc='upper right')
