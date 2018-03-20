import xgboost as xgb
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model_file = os.path.join(os.getcwd(), "model_weight/binary_xgboost/binary_xgboost.model")
    model = xgb.Boost(model_file = model_file)
    xgb.plot_importance(booster=model)
    plt.savefig("feature_importance.png", dpi=500)