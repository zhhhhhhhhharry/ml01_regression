import numpy as np

class linerRegressionGSD(object):
    '''手写朴素梯度下降法线性回归'''
    def __init__(self,learning_rate=0.01, n_estimators=100):
        self.learning_rate=learning_rate
        self.n_estimators=n_estimators
        self.w=None
        self.b=None
        self.costs=[]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range(self.n_estimators):

            y_hat = np.dot(X, self.w) + self.b
        
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            cost = np.mean((y_hat - y)**2)
            self.costs.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i}, cost: {cost:.4f}")
        
        return self
    
    def predict(self,X):
        y_hat=np.dot(X,self.w)+self.b
        return y_hat
    

class LinearRegressionSquare(object):
    '''最小二乘法线性回归'''
    def __init__(self):
        self.w = None  
        self.b = None  

    
    def fit(self, X, y):
        print("最小二乘法线性回归")
        X = np.insert(X, 0, 1, axis=1)
        try:
            X_transpose = np.transpose(X)
            self.w = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
        except np.linalg.LinAlgError:
            print("矩阵不可逆，无法进行线性回归")
            self.w = None
            self.b = None
            return

        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, X):
        # 在特征矩阵 X 的第一列添加一列全为1的列，以处理截距
        X = np.insert(X, 0, 1, axis=1)

        # 使用拟合的系数进行预测
        y_pred = X.dot(np.insert(self.w, 0, self.b))
        return y_pred
    

class LassoRegression(object):
    '''手写Lasso回归'''
    def __init__(self, learning_rate=0.01, alpha=0.1, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.costs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range(self.max_iter):
            y_hat = np.dot(X, self.w) + self.b
            dw = (2 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (2 / n_samples) * np.sum(y_hat - y)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            cost = np.mean((y_hat - y)**2)
            self.costs.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i}, cost: {cost:.4f}")
        
        return self
    
    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
class RidgeRegression(object):
    '''手写Ridge回归'''
    def __init__(self, learning_rate=0.01, alpha=0.1, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range(self.max_iter):
            y_hat = np.dot(X, self.w) + self.b
            dw = (2 / n_samples) * np.dot(X.T, (y_hat - y)) + 2 * self.alpha * self.w
            db = (2 / n_samples) * np.sum(y_hat - y)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            cost = np.mean((y_hat - y)**2) + self.alpha * np.sum(self.w**2)
            self.costs.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i}, cost: {cost:.4f}")
        
        return self
        
    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
class RandomForestRegressor: 
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []
    def fit(self, X, y):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.utils import resample
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)





#随机森林

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
# 节点类
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# 回归决策树
class RegressionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        # 样本数，特征数
        self.n_samples, self.n_features = X.shape
        # 确定最大特征数
        try:
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(self.n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(self.n_features))
            elif self.max_features == 'all':
                self.max_features = self.n_features
            elif isinstance(self.max_features, float):
                self.max_features = int(self.max_features * self.n_features)
                self.max_features = max(1, self.max_features)
        except:
            print("Invalid value for max_features. Allowed string values are 'sqrt', 'log2', 'all' or a float.")
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        # 停止条件
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
            # 叶节点值为目标变量的均值
            return Node(value=np.mean(y))
        # 随机选择特征
        rand_feat_seless = np.random.choice(self.n_features, self.max_features, replace=False)

        # 找到最佳分裂点
        best_feature, best_thresh = self._best_split(X, y, rand_feat_seless)

        # 分裂成左右子树
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        # 确保叶节点有足够样本
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return Node(value=np.mean(y))

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, rand_feat_seless):
        best_gain = -np.inf
        split_idx, split_thresh = None, None

        for rand_feat_sele in rand_feat_seless:
            X_column = X[:, rand_feat_sele]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # 计算分裂增益（方差减少量）
                gain = self._variance_reduction(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = rand_feat_sele
                    split_thresh = threshold
        return split_idx, split_thresh
    def _variance_reduction(self, y, X_column, threshold):
        # 父节点方差
        parent_var = np.var(y)
        # 分裂成左右子节点
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # 子节点方差
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        var_left, var_right = np.var(y[left_idxs]), np.var(y[right_idxs])
        # 计算方差减少量（信息增益）
        child_var = (n_left / n) * var_left + (n_right / n) * var_right
        return parent_var - child_var
    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# 随机森林回归器
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators  # 树的数量
        self.max_depth = max_depth  # 每棵树的最大深度
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.min_samples_leaf = min_samples_leaf  # 叶节点最小样本数
        self.max_features = max_features  # 每棵树考虑的最大特征数
        self.random_state = random_state  # 随机种子，保证可复现性
        self.trees = []  # 存储所有树
    # 新增：实现get_params方法，让GridSearchCV可以获取参数
    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state
        }

    # 新增：实现set_params方法，让GridSearchCV可以设置参数
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def fit(self, X, y):
        try:
            y = y.values.ravel()  # 确保是一维数组
        except:
            print("y must be a 1-dimensional array.")

        self.trees = []
        for _ in range(self.n_estimators):
            # 创建决策树
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )

            # Bootstrap抽样
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # 训练树
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        # 有放回抽样，生成与原样本量相同的数据集
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # 如果输入是DataFrame，转换为NumPy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        # 收集所有树的预测结果
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 回归问题：取所有树预测的平均值
        return np.mean(tree_preds, axis=0)

    def feature_importances_(self):
        """计算特征重要性"""
        importances = np.zeros(self.trees[0].n_features)
        for tree in self.trees:
            # 简单实现：统计每个特征被用作分裂点的次数
            self._calculate_tree_importance(tree.root, importances)
        # 归一化
        return importances / np.sum(importances)

    def _calculate_tree_importance(self, node, importances):
        if node.feature is not None:
            importances[node.feature] += 1
            self._calculate_tree_importance(node.left, importances)
            self._calculate_tree_importance(node.right, importances)

# 模型保存函数
def save_model(model, filename):
    """将训练好的模型保存到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到 {filename}")

if __name__ == "__main__":
    # 读取数据
    X = pd.read_csv('X_train.csv')
    # 去除Id列
    X = X.drop('Id', axis=1)
    y = pd.read_csv('y_train.csv')
    # 假设y的目标列名为'label'，如果不是请修改
    if 'Id' in y.columns:
        y = y.drop('Id', axis=1)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 转换为NumPy数组
    X_train = X_train.values
    X_test = X_test.values  # 新增测试集转换
    try:
        y_train = np.log(y_train)
        y_test = np.log(y_test)
    except:
        print("y must be a positive array.")
    # 训练随机森林回归模型
    #搜索网格
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,  # 3折交叉验证
        n_jobs=-1,  # 使用所有可用的CPU
        verbose=2,  # 输出搜索过程
        scoring='neg_mean_squared_error'  # 回归问题常用评分
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    # 保存模型
    save_model(best_model, 'rf_model.pkl')
