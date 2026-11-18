MODEL_KEYS = ["LogReg", "KNN", "RandomForest", "MLP", "LightGBM", "LightGBM_tuned22", "LightGBM_dir", "LightGBM_dir_tuned"]
DISPLAY = {
    "LogReg": "Logistic Regression",
    "KNN": "KNN (distance-weighted)",
    "RandomForest": "Random Forest",
    "MLP": "MLP",
    "LightGBM": "LightGBM",
    "LightGBM_tuned22": "LightGBM tuned (22 f)",
    "LightGBM_dir": "LightGBM + direction (29 f)",
    "LightGBM_dir_tuned": "LightGBM + direction, tuned (final)",
}
BASE_KEYS = ["LogReg", "KNN", "RandomForest", "MLP", "LightGBM"]
FINAL_KEY = "LightGBM_dir_tuned"
