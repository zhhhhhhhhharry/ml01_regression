import pandas as pd


def feature_selection_pca(data_processor, n_components=30):
    """特征选择"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)  # 保留30个主成分
    X_selected = pca.fit_transform(data_processor.x_Train.drop('Id', axis=1))
    X_selected = pd.DataFrame(X_selected, columns=[f'PC{i+1}' for i in range(X_selected.shape[1])])
    X_selected=X_selected[sorted(X_selected.columns)]
    X_selected.to_csv("data/selected/selected_x_train.csv", index=False)
    print(f"PCA降维后形状: {X_selected.shape}")
    # 提取测试集特征（删除Id列），用训练集拟合的PCA转换
    test_features = data_processor.x_Test.drop('Id', axis=1)
    X_selected_test = pca.transform(test_features)  # 使用训练集拟合的PCA转换测试集
    # 创建测试集主成分DataFrame并排序列（与训练集保持一致）
    X_selected_test = pd.DataFrame(
        X_selected_test, 
        columns=[f'PC{i+1}' for i in range(n_components)]  # 与训练集相同的列名
    )
    X_selected_test = X_selected_test[sorted(X_selected_test.columns)]  # 按相同规则排序列
    X_selected_test.to_csv("data/selected/selected_x_test.csv", index=False)  # 保存测试集结果
    data_processor.x_Test = X_selected_test  # 更新测试集为处理后的主成分数据
    print(f"测试集PCA后形状: {X_selected_test.shape}")
def feature_selection_rf(data_processor, n_features=30):
    from sklearn.ensemble import RandomForestRegressor
    # 训练随机森林模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(data_processor.x_Train.drop('Id', axis=1), data_processor.y_Train)
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': data_processor.x_Train.drop('Id', axis=1).columns,
        'importance': rf.feature_importances_
    })
    # 按重要性排序
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    print(feature_importance)
    # 选择前n个重要特征
    selected_features = feature_importance.head(n_features)['feature'].tolist()
    X_selected = data_processor.x_Train.drop('Id', axis=1)[selected_features]
    X_selected=X_selected[sorted(X_selected.columns)]
    X_selected.to_csv("data/selected/selected_x_train.csv", index=False)
    print(f"随机森林选择后形状: {X_selected.shape}")
    # 提取测试集特征（删除Id列），用训练集拟合的PCA转换
    test_features = data_processor.x_Test.drop('Id', axis=1)
    X_selected_test = test_features[selected_features]  
    X_selected_test=X_selected_test[sorted(X_selected_test.columns)]  
    X_selected_test.to_csv("data/selected/selected_x_test.csv", index=False)  
    data_processor.x_Test = X_selected_test 
    print(f"测试集相关系数选择后形状: {X_selected_test.shape}")


def feature_selection_correlation(data_processor, n_features=30):
    data_processor.x_Train['SalePrice']=data_processor.y_Train
    """基于相关系数的特征选择"""
    # 计算特征之间的相关系数矩阵
    corr_matrix = data_processor.x_Train.drop('Id', axis=1).corr().abs()
    # 选择相关系数大于阈值的特征
    selected_features = corr_matrix['SalePrice'].sort_values(ascending=False).head(n_features).index.tolist()
    X_selected = data_processor.x_Train.drop('Id', axis=1)[selected_features]
    X_selected=X_selected[sorted(X_selected.columns)]
    X_selected=X_selected.drop('SalePrice', axis=1)
    X_selected.to_csv("data/selected/selected_x_train.csv", index=False)
    print(f"相关系数选择后形状: {X_selected.shape}")
    # 提取测试集特征（删除Id列），用训练集拟合的PCA转换
    test_features = data_processor.x_Test.drop('Id', axis=1)
    selected_features.remove('SalePrice')  # 确保测试集中不包含目标变量
    X_selected_test = test_features[selected_features]  
    X_selected_test=X_selected_test[sorted(X_selected_test.columns)]  
    X_selected_test.to_csv("data/selected/selected_x_test.csv", index=False) 
    data_processor.x_Test = X_selected_test 
    print(f"测试集相关系数选择后形状: {X_selected_test.shape}")