print("Testing Python execution...")
import numpy as np
print("NumPy imported successfully")
from sklearn import datasets
print("Sklearn imported successfully")
data = datasets.fetch_california_housing()
print(f"Dataset loaded: {data.data.shape[0]} samples, {data.data.shape[1]} features")
print("Test completed successfully!")
