import os
import joblib 

import pandas as pd
import polars as pl
import numpy as np 
from tqdm import tqdm

import pickle

import torch
from sklearn.impute import SimpleImputer
from cuml.ensemble import RandomForestRegressor as cuRF
import cudf


def save_parquet(data, file_name):
    data.write_parquet(f'/kaggle/working/' + file_name + '.parquet', partition_by = "symbol_id")

# 离群值处理
class Outlier():
    def __init__(self, method = '3sig', all_columns = FEAT_COLS):
        self.method = method
        self.all_columns = all_columns
    def replace_outliers_3sigma(self, df):
        for column_name in self.all_columns:
            if df[column_name].is_empty():
                continue
            mean = df[column_name].mean()
            std_dev = df[column_name].std()
            lower_bound = mean - 3 * std_dev
            upper_bound = mean + 3 * std_dev
            df = df.with_columns(
                pl.when(pl.col(column_name) < lower_bound)
                .then(pl.lit(lower_bound))
                .when(pl.col(column_name) > upper_bound)
                .then(pl.lit(upper_bound))
                .otherwise(pl.col(column_name))
                .alias(column_name)
            )
        return df
    
    def replace_outliers_iqr(self, df):
        for column_name in self.all_columns:
            if df[column_name].null_count() == df[column_name].shape[0]:
                continue
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df.with_columns(
                pl.when(pl.col(column_name) < lower_bound)
                .then(pl.lit(lower_bound))
                .when(pl.col(column_name) > upper_bound)
                .then(pl.lit(upper_bound))
                .otherwise(pl.col(column_name))
                .alias(column_name)
            )
        return df

    def preprocesing(self, df):
        if self.method == '3sig':
            return self.replace_outliers_3sigma(df)
        else:
            return self.replace_outliers_iqr(df)

# 随机缺失填补
class FillRandomMissing:
    def __init__(self, df, method = 'mean', columns = FEAT_COLS, threshold=10):
        self.method = method
        self.col = columns
        self.threshold = threshold
        self.data = df[TIME_COLS + LEAD_COLS + [self.col]]
        
        
    def interpolated(self,data):
        return data.with_columns(pl.col(self.col).interpolate())

    def arima(self, data, param = (1,1,1)):
        pass
        
    def random_missing(self):
        col=self.col
        groups = self.data.group_by('symbol_id')
        all_random_missing = pl.DataFrame()
        for k, d in groups:
            d = d.with_columns(pl.col(col).is_not_null().cum_sum().alias("not_null"))
            data_missing  = d['not_null']
            missing_count = data_missing.value_counts()
            missing_count_lst = missing_count.filter(pl.col("count") < self.threshold)['not_null'].to_list()
            random_missing = self.interpolated(d.filter(pl.col('not_null').is_in(missing_count_lst)))
            all_random_missing = pl.concat([all_random_missing, random_missing], how="vertical")
        return all_random_missing[TIME_COLS + LEAD_COLS + [col]] # 连续非空小于threshold的部分

class FillSystemMissing:
    def __init__(self, df, feature,buffer = 0.5,
                 corr_data_file = f'/kaggle/input/corr-data/corr_pearson.csv',
                 model_path = f'/kaggle/input/randomforest-filldata-model'):
        self.corr_data = pd.read_csv(corr_data_file, index_col = 0)
        self.missing_data = df[FEAT_COLS].null_count()
        self.buffer = buffer
        self.data = df[TIME_COLS + LEAD_COLS + self.corr_data.index[abs(self.corr_data[feature])>self.buffer].to_list()]
        self.model_path = model_path
        self.feature = feature
        
    def get_model(self):
        with open(self.model_path + f'/'+ self.feature +f'_model.pkl', 'rb') as file:
            # 加载.pkl文件中的对象
            model = pickle.load(file)
        return model
    def fill_data(self,):

        model = self.get_model()
        mask = self.data[self.feature].is_null()
        
        # 然后，我们选择这些行的其他列的值
        rows_to_predict = self.data.filter(mask).drop(TIME_COLS + LEAD_COLS + [self.feature])
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        rows_to_predict = imputer.fit_transform(rows_to_predict)

        predicted_values = []
        for i in range(rows_to_predict.shape[0]):
            try:
                predicted_values.append(model.predict(rows_to_predict[i].reshape(1,-1))[0])
            except:
                predicted_values.append(0)
        temp = self.data[TIME_COLS + LEAD_COLS ].filter(mask)
        temp = temp.with_columns(pl.Series('p',predicted_values).alias(self.feature+'fill'))
        temp_df = self.data[TIME_COLS + LEAD_COLS + [self.feature]].join(temp, on = TIME_COLS + LEAD_COLS, how = 'left')
        del temp, predicted_values
        temp_df = temp_df.with_columns(pl.col(self.feature).fill_null(0).alias(self.feature))
        temp_df = temp_df.with_columns(pl.col(self.feature + 'fill').fill_null(0).alias(self.feature+'fill'))
        temp_df = temp_df.with_columns((pl.col(self.feature)+pl.col(self.feature + 'fill')).alias(self.feature))
        
        return temp_df[self.feature]

class FillAllData():
    def __init__(self,
                model_path = f'/kaggle/input/randomforest-filldata-model'):
        file_names = os.listdir(model_path)
        # 过滤掉文件夹，只保留文件
        self.system_features = [f[: -len('_model.pkl')] for f in file_names if os.path.isfile(os.path.join(model_path, f)) and f[:len('feature')] == 'feature']
        
    def fill_all_data(self, df):
        df = Outlier(method = 'iqr', all_columns = FEAT_COLS).preprocesing(df)
        # self.df = FillSystemMissing(self.df).fill_data()
        for sf in tqdm(self.system_features):
            temp = FillSystemMissing(df, sf,buffer = 0.3)
            df = df.with_columns(temp.fill_data().alias(sf))
            del temp
        for f in tqdm(FEAT_COLS):
            try:
                df = df.with_columns(df[f].fill_null(df[f].mean()).alias(f))
            except:
                df = df.with_columns(df[f].fill_null(0).alias(f))
        
        return df


          
