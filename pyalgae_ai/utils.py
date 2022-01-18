import pandas as pd

# Date 형 변환
def set_date(rawframe):
    rawframe['set_date'] = pd.to_datetime(rawframe[rawframe.columns[0]], format='%Y%m%d %H:%M', errors='coerce')
    rawframe = rawframe.drop(rawframe.columns[0], axis=1)
    return rawframe


# DataFrame merge
def set_merge(tb1, tb2):
    raw_return = pd.merge(tb1, tb2, how='outer', on='set_date')
    return raw_return


# 데이터 결측 범위 지정 결측값 도출
def set_outlier(df1):
    columns = list(range(1, df1.shape[1]))
    for i in columns:
        df1[df1.columns[i]].mask(
            (df1[df1.columns[i]] >= 999) | (df1[df1.columns[i]] < -30), inplace=True)
    return df1