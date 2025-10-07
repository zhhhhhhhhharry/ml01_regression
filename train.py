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
        print("最小二乘法线性回归")
        model = models.LinearRegressionSquare()
    elif modelName == "lasso":
        model = models.LassoRegression(learning_rate=args.lr, alpha=args.alpha, max_iter=args.n_estimators)
    elif modelName == "ridge":
        model = models.RidgeRegression(learning_rate=args.lr, alpha=args.alpha, max_iter=args.n_estimators)
    elif modelName == "rf":
        model = models.RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    else:
        raise ValueError("Invalid model name. Please choose from 'naive', 'linear', 'lasso', 'ridge', 'rf', 'xgb', 'lgb'.")
    return model


def model_selection_skl(modelName):
    if modelName == "naive":
        model = models.linerRegressionGSD(learning_rate=args.lr, n_estimators=args.n_estimators)
    elif modelName == "linear":
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor(eta0=args.lr, max_iter=args.n_estimators)
    elif modelName == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=args.alpha)
    elif modelName == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=args.alpha)
    elif modelName == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
    elif modelName == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    else:
        raise ValueError("Invalid model name. Please choose from 'naive', 'linear', 'lasso', 'ridge', 'rf', 'xgb'.")
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



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def model_train_kfold_regression(model, X, y, n_splits=5, random_state=42):
    """
    使用K折交叉验证训练回归模型，使用RMSE和R²评估
    
    参数:
    model: 回归模型
    X: 特征数据
    y: 目标变量
    n_splits: K折数，默认为5
    random_state: 随机种子
    
    返回:
    results_df: 每折的详细结果DataFrame
    overall_metrics: 总体评估指标
    y_test_all: 所有测试集的真实值
    y_pred_all: 所有测试集的预测值
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 存储每折的结果
    fold_results = []
    y_test_all = []
    y_pred_all = []
    
    print("K-Fold Cross Validation Results (Regression)")
    print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if args.way == "sklearn":
            model=model_selection_skl(args.model)
        else:
            model=model_selection_div(args.model)
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))  # 可选：平均绝对误差
        
        # 存储该折结果
        fold_results.append({
            'Fold': fold,
            'RMSE': rmse,
            'R2_Score': r2,
            'MAE': mae,
            'Train_Size': len(X_train),
            'Test_Size': len(X_test)
        })
        
        # 存储所有预测结果
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)
        
        print(f"Fold {fold}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(fold_results)
    
    # 转换为numpy数组
    y_test_all = np.array(y_test_all)
    y_pred_all = np.array(y_pred_all)
    
    # 计算总体指标
    overall_rmse = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
    overall_r2 = r2_score(y_test_all, y_pred_all)
    overall_mae = np.mean(np.abs(y_test_all - y_pred_all))
    
    overall_metrics = {
        'Overall_RMSE': overall_rmse,
        'Overall_R2': overall_r2,
        'Overall_MAE': overall_mae,
        'Mean_CV_RMSE': results_df['RMSE'].mean(),
        'Std_CV_RMSE': results_df['RMSE'].std(),
        'Mean_CV_R2': results_df['R2_Score'].mean(),
        'Std_CV_R2': results_df['R2_Score'].std()
    }
    
    # 打印总结
    print("\n" + "=" * 60)
    print("K-Fold Cross Validation Summary (Regression)")
    print("=" * 60)
    print(results_df.round(4))
    
    print("\nOverall Metrics:")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print(f"Overall R²: {overall_r2:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Mean CV RMSE: {overall_metrics['Mean_CV_RMSE']:.4f} (±{overall_metrics['Std_CV_RMSE']:.4f})")
    print(f"Mean CV R²: {overall_metrics['Mean_CV_R2']:.4f} (±{overall_metrics['Std_CV_R2']:.4f})")
    
    return results_df, overall_metrics, y_test_all, y_pred_all

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
    elif args.feature == "cor_pca":
       ds.feature_selection_cor_pca(data_processor, n_features=210, n_components=args.dim)
    else:
        print("Using full feature set.")

    # 选择模型
    if args.way == "sklearn":
        model = model_selection_skl(args.model)
    else:
        model = model_selection_div(args.model)

    #训练模型
    print("Training model...")
    #x_Train=pd.read_csv('data/selected/selected_x_train.csv')
    #data_processor.x_Train=pd.read_csv('data/preprocess/x_train.csv')
    # data_processor.y_Train=pd.read_csv('data/preprocess/y_train.csv')
    # y_test, y_pred = model_train(model, data_processor.x_Train, data_processor.y_Train)
    # print("Model trained.")
    # print("Evaluating model...")

    #k折
    results_df, overall_metrics, y_test_all, y_pred_all =model_train_kfold_regression(model, data_processor.x_Train, data_processor.y_Train, n_splits=5)
   
    for y_test, y_pred in [(y_test_all, y_pred_all)]:
        model_evaluation(y_test, y_pred)

    # # 评估模型
    # model_evaluation(y_test, y_pred)


    


    # # 预测
    # print("Making predictions...")
    # predictions = model.predict(data_processor.x_Test.drop('Id', axis=1))

    # 保存结果
#    submission(model)



    #wait to be done
    #数据预分析、去离散值
    #k折验证
    #超参数调优（降维维度、学习率、正则化参数）
    #剩余model手写
    #结果可视化（损失曲线、真实值与预测值对比图）
    #集成模型（简单平均、加权平均、堆叠）--可选


