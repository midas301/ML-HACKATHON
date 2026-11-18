from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

SEED = 42

LGBM_DEFAULT = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                    subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, random_state=SEED, n_jobs=2, verbose=-1)

REGISTRY = {
    "LogReg": ("Linear / probabilistic", "StandardScaler + L2 logistic regression, C=1.0, lbfgs, max_iter=500",
               lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)), None),
    "KNN": ("Instance-based", "StandardScaler + distance-weighted KNN, k=25, trained on 150k-row subsample",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2)), 150000),
    "RandomForest": ("Ensemble (bagging)", "200 trees, min_samples_leaf=5, max_features=sqrt, max_samples=0.3",
                     lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_samples=0.3, n_jobs=2, random_state=SEED), None),
    "LightGBM": ("Boosting", "LightGBM, 300 trees, lr=0.05, num_leaves=31, min_child_samples=20",
                 lambda: LGBMClassifier(**LGBM_DEFAULT), None),
    "MLP": ("Neural network", "StandardScaler + MLP(64,32), ReLU, Adam lr=1e-3, alpha=1e-4, batch=1024, 25 epochs",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                                                  learning_rate_init=1e-3, max_iter=25, random_state=SEED)), None),
}
