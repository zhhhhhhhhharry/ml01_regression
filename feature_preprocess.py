#导入相关库
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
#导入数据集

class DataPreprocess:
    def __init__(self):
        #初始化
        self.sample_submission = None
        self.train_data = None
        self.test_data = None
        self.total_data = None
        self.col_missing = None
        self.missing_cols = None
        self.character_cols = None
        self.x_Train = None
        self.x_Test = None
        self.y_Train = None
        self.y_Test = None
        self.new_total_data = None
        self.ordinal_features = {
            'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],                   # 地块形状
            'Utilities': ['NA','ELO', 'NoSeWa', 'NoSewr', 'AllPub'],         # 公用设施
            'LandSlope': ['Sev', 'Mod', 'Gtl'],                         # 地块坡度
            'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],               # 外部质量
            'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],               # 外部状况
            'BsmtQual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],          # 地下室质量
            'BsmtCond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],          # 地下室状况
            'BsmtExposure': ['NA', 'No', 'Mn', 'Av', 'Gd'],            # 地下室采光
            'BsmtFinType1': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'BsmtFinType2': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],               # 供暖质量
            'Electrical': ['NA','FuseP', 'FuseF', 'FuseA', 'Mix', 'SBrkr'], # 电气系统
            'KitchenQual': ['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'],             # 厨房质量
            'Functional': ['NA','Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
            'FireplaceQu': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],       # 壁炉质量
            'GarageFinish': ['NA', 'Unf', 'RFn', 'Fin'],               # 车库装修
            'GarageQual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],        # 车库质量
            'GarageCond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],        # 车库状况
            'PavedDrive': ['N', 'P', 'Y'],                             # 铺砌车道
            'PoolQC': ['NA', 'Fa', 'TA', 'Gd', 'Ex'],                  # 游泳池质量
            'Fence': ['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],         # 围栏质量
        }
    def load_data(self):
        #加载数据集
        self.sample_submission = pd.read_csv('data/raw/sample_submission.csv',encoding='gbk')
        self.train_data = pd.read_csv('data/raw/train.csv',encoding='gbk')
        #self.train_data = self.train_data.drop(self.train_data[self.train_data['Id'] == 1299].index)
        #self.train_data = self.train_data.drop(self.train_data[self.train_data['Id'] == 524].index)
        self.test_data = pd.read_csv('data/raw/test.csv',encoding='gbk')

        self.split_data()
        self.total_data = pd.concat([self.x_Train, self.x_Test], axis=0)
        self.columns_encoding = [col for col in self.total_data.columns.tolist() if col != 'Id']
    def split_data(self):
        # 划分数据集
        self.x_Train = self.train_data.drop('SalePrice', axis=1)
        self.y_Train = self.train_data['SalePrice']
        self.x_Test = self.test_data
        self.y_Test = None
    def lookup_data(self):
        #查看数据集信息
        print(self.sample_submission.head())
        print(self.x_Train.head())
        print(self.x_Test.head())
        #查看每一行缺失值，全部打印出来
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns',None)
        self.col_missing = self.total_data.isnull().sum(axis=0)
        #寻找存在缺失值列的标签
        self.missing_cols = self.col_missing[self.col_missing > 0].index
        print(self.col_missing)
        print(self.missing_cols)
    def handle_missing_data(self):
        """处理缺失值"""
        for col in self.missing_cols:
            if col != 'Id' :  # 避免Id特征被填充
                if self.total_data[col].dtype == 'int64' or self.total_data[col].dtype == 'float64':
                    #如果超过10%的样本缺失，直接删除该列
                    if self.col_missing[col]>0.1*len(self.total_data):
                        self.total_data.drop(col,axis=1,inplace=True)
                    else:
                        self.total_data[col] = self.total_data[col].fillna(self.total_data[col].median())
                else:
                    self.total_data[col] = self.total_data[col].fillna('NA')
        self.x_Train=self.total_data[:len(self.x_Train)]
        self.x_Test=self.total_data[len(self.x_Train):]
    def normalize_features(self):
        """对数值型特征进行标准化"""
        numeric_cols = self.total_data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'Id' :  # 避免Id特征被标准化
                scaler = StandardScaler()
                self.total_data[col] =scaler.fit_transform(self.total_data[col].values.reshape(-1, 1))
                self.x_Train=self.total_data[:len(self.x_Train)]
                self.x_Test=self.total_data[len(self.x_Train):]
        for col in numeric_cols:
            feature_sk = self.x_Train[col].skew()
            if feature_sk > 1:
                if (self.x_Train[col] > 0).all() and (self.x_Test[col] > 0).all():
                    print(f"正偏态特征 {col} 存在负值或零值")
                    self.x_Train[col] = np.log1p(self.x_Train[col])
                    self.x_Test[col] = np.log1p(self.x_Test[col])
                    self.total_data[col] = np.log1p(self.total_data[col])
                else:
                    self.x_Train[col] = np.sign(self.x_Train[col]) * np.log1p(np.abs(self.x_Train[col]))
                    self.x_Test[col] = np.sign(self.x_Test[col]) * np.log1p(np.abs(self.x_Test[col]))
                    self.total_data[col] = np.sign(self.total_data[col]) * np.log1p(np.abs(self.total_data[col]))
            elif feature_sk < -1:
                print(f"负偏态特征 {col} 存在负值或零值")
                self.x_Train[col] = self.x_Train[col] ** 2
                self.x_Test[col] = self.x_Test[col] ** 2
                self.total_data[col] = self.total_data[col] ** 2
        self.y_Train = np.log1p(self.y_Train)  # 对目标变量取对数

    def string_encoding(self):
        for col in self.ordinal_features:
            if col in self.total_data.columns:
                # 应用映射
                self.total_data[col] = self.total_data[col].map(
                    {k: v for v, k in enumerate(self.ordinal_features[col])}
                )
                scaler = StandardScaler()
                self.total_data[col] =scaler.fit_transform(self.total_data[col].values.reshape(-1, 1))
        # 获取所有分类特征列（排除Id）
        self.character_cols = [col for col in self.total_data.dtypes[self.total_data.dtypes == 'object'].index
                               if col != 'Id']
        if not self.character_cols:
            return
        #若果该列只有NA和另一个值，则对该列进行0,1编码
        # 遍历副本以避免遍历中修改列表导致的问题
        # for col in list(self.character_cols):
        #     if len(self.total_data[col].unique()) == 2:
        #         # 对单列进行独热编码并合并回原数据集
        #         dummies = pd.get_dummies(self.total_data[col], prefix=col, drop_first=True)  # drop_first=True 实现二元编码
        #         self.total_data = pd.concat([self.total_data, dummies], axis=1)  # 合并编码后的列
        #         self.total_data.drop(col, axis=1, inplace=True)  # 删除原始列
        #         self.character_cols.remove(col)  # 从字符列列表中移除

                # 创建一个编码器实例
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'可避免多重共线性
        # 对所有分类特征进行编码
        encoded_data = encoder.fit_transform(self.total_data[self.character_cols])
        # 获取编码后的特征名称
        encoded_feature_names = encoder.get_feature_names_out(self.character_cols)
        # 创建编码后的DataFrame
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_feature_names,
            index=self.total_data.index
        )
        # 删除原始分类列，拼接编码后的列
        self.total_data = self.total_data.drop(columns=self.character_cols)
        self.new_total_data = pd.concat([self.total_data, encoded_df], axis=1)
        # 重新划分训练集和测试集
        train_size = len(self.x_Train)
        self.x_Train = self.new_total_data.iloc[:train_size]
        self.x_Test = self.new_total_data.iloc[train_size:]
    def save_data(self):
        """保存处理后的数据"""
        self.x_Train.to_csv('data/preprocess/x_train.csv',index=False)
        self.x_Test.to_csv('data/preprocess/x_test.csv',index=False)
        self.y_Train.to_csv('data/preprocess/y_train.csv',index=False)
        #记录处理后的列名
        self.columns_new_encoding = [col for col in self.new_total_data.columns.tolist() if col != 'Id']
        with open('columns_encoding.txt','w') as f:
            f.write(str(self.columns_encoding))
        with open('columns_new_encoding.txt','w') as f:
            f.write(str(self.columns_new_encoding))

    def data_change(self):
        for col in self.x_Train.columns:
            feature_sk = self.x_Train[col].skew()
            # 对正偏态数据使用对数变换
            if feature_sk > 1:
                # 确保数据都是正值（对数变换要求）
                if (self.x_Train[col] > 0).all() and (self.x_Test[col] > 0).all():
                    self.x_Train[col] = np.log1p(self.x_Train[col])
                    self.x_Test[col] = np.log1p(self.x_Test[col])
                else:
                    # 如果有负值或零值，使用其他变换
                    self.x_Train[col] = np.sign(self.x_Train[col]) * np.log1p(np.abs(self.x_Train[col]))
                    self.x_Test[col] = np.sign(self.x_Test[col]) * np.log1p(np.abs(self.x_Test[col]))

            # 对负偏态数据使用平方变换或其他适当变换
            elif feature_sk < -1:
                # 平方变换可以减轻负偏态
                self.x_Train[col] = self.x_Train[col] ** 2
                self.x_Test[col] = self.x_Test[col] ** 2