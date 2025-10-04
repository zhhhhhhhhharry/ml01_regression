import args as args
import feature_preprocess as fp
import dimension_selection as ds
import models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



#"naive", "linear", "lasso", "ridge", "rf", "xgb", "lgb"
def model_selection_div(modelName):
    if modelName == "naive":
        model = models.linerRegressionGSD(learning_rate=args.lr, n_estimators=args.n_estimators)
    elif modelName == "linear":
        model = models.LinearRegressionSquare()
    elif modelName == "lasso":
        model = models.LinearRegressionLasso()
    elif modelName == "ridge":
        model = models.LinearRegressionRidge()
    elif modelName == "rf":
        model = models.RandomForestRegressor()
    else:
        raise ValueError("Invalid model name. Please choose from 'naive', 'linear', 'lasso', 'ridge', 'rf', 'xgb', 'lgb'.")
    return model


def model_selection_skl(modelName):
    if modelName == "naive":
        model = models.linerRegressionGSD(learning_rate=args.lr, n_estimators=args.n_estimators)
    elif modelName == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif modelName == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=args.alpha)
    elif modelName == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge()
    elif modelName == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
    elif modelName == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    elif modelName == "lgb":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor()
    else:
        raise ValueError("Invalid model name. Please choose from 'naive', 'linear', 'lasso', 'ridge', 'rf', 'xgb', 'lgb'.")
    return model

def model_train(model, X, y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred

def model_evaluation(y_test, y_pred):
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print("MSE:",mse)
    print("RMSE:",rmse)
    print("MAE:",mae)
    print("R2:",r2)
    plt.scatter(y_test,y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True Values vs Predictions")
    plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
    plt.show()


def submission(model):
    x_Test = pd.read_csv('data/selected/selected_x_test.csv')
    predictions = model.predict(x_Test)
    predictions = np.array(predictions, dtype=np.float64)
    predictions=np.expm1(predictions)
    id=pd.read_csv('data/raw/test.csv',encoding='gbk')['Id']
    submission = pd.DataFrame({
        'Id': id,
        'SalePrice': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    args = args.get_args()
    print("Arguments:", args)
    
    # 数据预处理
    data_processor = fp.DataPreprocess()
    data_processor.load_data()
    data_processor.lookup_data()
    data_processor.handle_missing_data()
    data_processor.normalize_features()
    data_processor.string_encoding()
    data_processor.save_data()

    # 特征选择
    if args.feature == "pca":
        ds.feature_selection_pca(data_processor, n_components=args.dim)
    elif args.feature == "rf":
        ds.feature_selection_rf(data_processor, n_features=args.dim)
    elif args.feature == "cov":
        ds.feature_selection_correlation(data_processor, n_features=args.dim)
    else:
        print("Using full feature set.")

    # 选择模型
    if args.way == "sklearn":
        model = model_selection_skl(args.model)
    else:
        model = model_selection_div(args.model)

    # 训练模型
    print("Training model...")
    x_Train=pd.read_csv('data/selected/selected_x_train.csv')
    data_processor.x_Train=x_Train
    y_test, y_pred = model_train(model, data_processor.x_Train, data_processor.y_Train)
    print("Model trained.")
    print("Evaluating model...")


    # 评估模型
    model_evaluation(y_test, y_pred)

    # # 预测
    # print("Making predictions...")
    # predictions = model.predict(data_processor.x_Test.drop('Id', axis=1))

    # 保存结果
    submission(model)
