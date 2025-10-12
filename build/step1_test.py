# Step 1: Import the required packages
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("SUCCESS: All packages imported successfully!")
print("Package versions:")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
import sklearn
print(f"Scikit-learn: {sklearn.__version__}")
print("Matplotlib and Seaborn styling configured!")
print("\nReady to proceed to Step 2!")
