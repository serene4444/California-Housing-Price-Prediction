# Import the required packages
import numpy as np
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

# Load the housing dataset and shuffle the data so that you don't bias your analysis.
data = datasets.fetch_california_housing()
X, y = shuffle(data.data, data.target, random_state=7)
# Use a smaller subset for faster training
X = X[:5000]  # Use only first 5000 samples
y = y[:5000]

# Split the dataset into training and testing in an 80/20 format:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Create and train the Support Vector Regressor using a linear kernel.
print("Training SVR model...")
sv_regressor = SVR(kernel='linear', C=1.0, max_iter=1000)  # Add max_iter for faster training
sv_regressor.fit(X_train, y_train)
print("Training completed!")

# Run the regressor on the testing data and predict the output (predicted labels).
y_test_pred = sv_regressor.predict(X_test)

# Evaluate the performance of the regressor and print the initial metrics.
print("Mean squared error: ", mean_squared_error(y_test, y_test_pred))
print("Explained variance score: ", explained_variance_score(y_test, y_test_pred))

# binarize the predicted values & the actual values using threshold of 25.00.
threshold = 2.0
y_pred_label = (y_test_pred > threshold).astype(int)
y_test_label = (y_test > threshold).astype(int)

# Create the confusion matrix using the predicted labels and the actual labels.
confusion_mat = confusion_matrix(y_test_label, y_pred_label)

# Visualize the confusion matrix.
print("\nConfusion Matrix:")
print(confusion_mat)
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange (2)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()


# Print the classification report based on the confusion matrix.
print(classification_report(y_test_label, y_pred_label))
print(f"\nThreshold: 2.0")
print(f"Actual labels - Class 0: {sum(y_test_label == 0)}, Class 1: {sum(y_test_label == 1)}")
print(f"Predicted labels - Class 0: {sum(y_pred_label == 0)}, Class 1: {sum(y_pred_label == 1)}")