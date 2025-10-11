import pickle
import pandas as pd
import numpy as np
# 从手写的model.py中导入自定义的随机森林回归器类
from rand import RandomForestRegressor, RegressionTree, Node
# 模型加载函数
def load_model(filename):
    """从文件加载模型"""
    try:
        with open(filename, 'rb') as f:
            rf_model = pickle.load(f)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None
    return rf_model

if __name__ == '__main__':
    rf_model = load_model('rf_model.pkl')
    try:
        new_data = pd.read_csv('selected_cor_x_test.csv')
        if 'Id' in new_data.columns:
            new_ids = new_data['Id']
            new_features = new_data.drop('Id', axis=1)
        else:
            new_ids = np.arange(len(new_data))
            new_features = new_data
        new_features = new_features.values

        predictions = rf_model.predict(new_features)

        predictions = np.exp(predictions)  # 还原为原始尺度

        result_df = pd.DataFrame({
            'Id': new_ids+1461,
            'SalePrice': predictions
        })
        result_df.to_csv('prices_pred.csv', index=False)
        print("新数据的预测结果已保存到 prices_pred.csv")
    except Exception as e:
        print(f"预测过程出错: {str(e)}")