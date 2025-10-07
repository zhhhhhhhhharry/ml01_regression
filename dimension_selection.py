import pandas as pd


def feature_selection_pca(data_processor, n_components=30):
    """特征选择"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)  # 保留30个主成分
    if 'Id' in data_processor.x_Train.columns:
        data_processor.x_Train = data_processor.x_Train.drop('Id', axis=1)
    if 'Id' in data_processor.x_Test.columns:
        data_processor.x_Test = data_processor.x_Test.drop('Id', axis=1)
    X_selected = pca.fit_transform(data_processor.x_Train)
    X_selected = pd.DataFrame(X_selected, columns=[f'PC{i+1}' for i in range(X_selected.shape[1])])
    #X_selected=X_selected[sorted(X_selected.columns)]
    data_processor.x_Train = X_selected
    X_selected.to_csv("data/selected/selected_x_train.csv", index=False)
    print(f"PCA降维后形状: {X_selected.shape}")
    # 提取测试集特征（删除Id列），用训练集拟合的PCA转换
    if 'Id' in data_processor.x_Test.columns:
        data_processor.x_Test = data_processor.x_Test.drop('Id', axis=1)
    X_selected_test = pca.transform(data_processor.x_Test)  # 使用训练集拟合的PCA转换测试集
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
    #X_selected=X_selected[sorted(X_selected.columns)]
    data_processor.x_Train = X_selected
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
   # X_selected=X_selected[sorted(X_selected.columns)]
    X_selected=X_selected.drop('SalePrice', axis=1)
    data_processor.x_Train = X_selected
    X_selected.to_csv("data/selected/selected_x_train.csv", index=False)
    print(f"相关系数选择后形状: {X_selected.shape}")
    # 根据特征数量自适应调整大小
    # 准备热力图数据
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # features_to_plot = [f for f in selected_features if f != 'SalePrice']
    # features_to_plot.append('SalePrice')
    # corr_subset = corr_matrix.loc[features_to_plot, features_to_plot]
    
    # # 自适应图形大小
    # fig_size = max(12, len(features_to_plot) * 0.9)
    # plt.figure(figsize=(fig_size, fig_size))
    
    # # 绘制热力图
    # sns.heatmap(corr_subset,
    #             annot=True,
    #             fmt=".2f",
    #             cmap='coolwarm',
    #             square=True,
    #             center=0,
    #             annot_kws={'size': 9},
    #             cbar_kws={'shrink': 0.8},
    #             linewidths=0.5)
    
    # plt.title(f"Selected Features Correlation Matrix\n(Top {n_features-1} Features)", 
    #             fontsize=14, fontweight='bold', pad=20)
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.show()

    # 提取测试集特征（删除Id列），用训练集拟合的PCA转换
    test_features = data_processor.x_Test.drop('Id', axis=1)
    selected_features.remove('SalePrice')  # 确保测试集中不包含目标变量
    X_selected_test = test_features[selected_features]  
    X_selected_test=X_selected_test[sorted(X_selected_test.columns)]
    data_processor.x_Test = X_selected_test  
    X_selected_test.to_csv("data/selected/selected_x_test.csv", index=False) 
    data_processor.x_Test = X_selected_test 
    print(f"测试集相关系数选择后形状: {X_selected_test.shape}")



def feature_selection_cor_pca(data_processor, n_features, n_components):
    """基于相关系数和PCA的特征选择"""
    feature_selection_correlation(data_processor, n_features)
    feature_selection_pca(data_processor, n_components)