import argparse

def get_args():
    parser = argparse.ArgumentParser(description="House Prices Regression Experiments")
    
    # 数据处理
    parser.add_argument("--feature", type=str, default="rf",
                        choices=["full", "pca", "rf", "cov"],
                        help="Feature type: full / pca / rf / cov")
    
    # 模型
    parser.add_argument("--model", type=str, default="xgb",
                        choices=["naive", "linear", "lasso", "ridge", "rf", "xgb", "lgb"],
                        help="Model type")
    
    #实现方式
    parser.add_argument("--way",type=str,default="sklearn",
                        choices=["sklearn","diy"],
                        help="Implementation way: sklearn / diy")
    
    # 降维维度
    parser.add_argument("--dim", type=int, default=30,
                        help="Dimension for PCA or covariance-based reduction")
    
    # 是否超参调优
    parser.add_argument("--tune", action="store_true",
                        help="Enable hyperparameter tuning")
    
    # 通用超参数
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Regularization strength (Ridge/Lasso)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for boosting models")
    parser.add_argument("--n_estimators", type=int, default=3000,
                        help="Number of trees / training rounds")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Max depth for tree-based models")
    
    return parser.parse_args()