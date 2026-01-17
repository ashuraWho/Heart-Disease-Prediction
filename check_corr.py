import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from notebooks.shared_utils import ARTIFACTS_DIR, setup_environment

setup_environment()

# Load processed data
X = np.load(ARTIFACTS_DIR / "X_train.npz")["X"]
y = np.load(ARTIFACTS_DIR / "y_train.npy")

# Convert to DF for corr
df = pd.DataFrame(X)
df["Target"] = y

print("Feature Correlations with Target:")
corrs = df.corr()["Target"].sort_values(ascending=False)
print(corrs)
