import numpy as np

class linerRegressionGSD(object):
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
        
            dw = (2 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (2 / n_samples) * np.sum(y_hat - y)
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
    def __init__(self):
        self.w = None  
        self.b = None  

    
    def fit(self, X, y):
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