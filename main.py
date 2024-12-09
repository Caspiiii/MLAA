from sklearn import datasets
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import dataprocessing as dp
import matplotlib.pyplot as plt
import pruning
import json
from sklearn.linear_model import LogisticRegression

########################################################################################################################
##
##                      -- MAIN FILE --
##  Entry point for executing training and prediction based pruning with SVM/SVR.
##
########################################################################################################################


## Deal with data being heavily inbalanced (a lot more negative)
def sparsify(input, size):
    """Sparsify the input down to the size given in size"""
    num_to_select = size
    selected_elements = np.random.choice(input, num_to_select, replace=False)
    return selected_elements


###############################################################################################################################################################################################################
##          Regression
###############################################################################################################################################################################################################
print("################################################################################################################################################################################################################################")
print("## Regression")
print("################################################################################################################################################################################################################################")


"""

inputVector = dp.create_input('l1/')

y = []
X = []
length = len(inputVector[0])
print(len(inputVector))
print(len(inputVector[0]))
for j in range(len(inputVector)):
    for i in range(len(inputVector[j])):
        if (i+1) % length == 0:
            y.append(inputVector[j][i])


for i in range(len(inputVector)):
    dataPoint = []
    for j in range(len(inputVector[i])):
        if (j + 1) % length != 0:
            dataPoint.append(inputVector[i][j])
        if ((j + 1) % length == 0):
            X.append(dataPoint)
            dataPoint = []
print(np.array(y).shape)
#print(X)
yNp = np.array(y)
XNp = np.array(X)
yPositive = yNp[yNp > 0]
xPositive = XNp[yNp > 0]
yNpNegative = yNp[yNp <= 0]
xNpNegative = XNp[yNp <= 0]
yNegative = sparsify(yNpNegative, len(xPositive))
xNegative = xNpNegative[np.random.choice(xNpNegative.shape[0], len(xPositive), replace=False)]
X = np.vstack((xNegative, xPositive))
y = np.hstack((yNegative, yPositive))
print("Y: ")
print(y)
print(y)



for i in range(length-1):
    plt.scatter(y,  X[:,i], label='Actual Data')
    plt.xlabel('Indices')
    plt.ylabel('Feature')
    plt.title('SVR Predictions vs Actual')
    plt.legend()
    plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVR model with RBF kernel
svr = SVR(kernel='rbf')

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

print("Not yet fitted")

# Initialize GridSearchCV
grid_search = GridSearchCV(svr, param_grid, cv=5, verbose = 3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Fitted")

# Best parameters and model
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
best_svr = grid_search.best_estimator_

y_pred = best_svr.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# Save the model to a file
joblib_file = "models/svr_model_one_time_metric.pkl"
joblib.dump(best_svr, joblib_file)

# Scatter plot of actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR Predictions vs Actual')
plt.legend()
plt.show()

# Individual feature plot
for i in range(X_test.shape[1]):
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test[:, i], y_test, color='black', label='Actual Data')
    plt.scatter(X_test[:, i], y_pred, color='red', label='Predicted Data')
    plt.xlabel(f'Feature {i}')
    plt.ylabel('Target')
    plt.title(f'SVR Predictions vs Actual for Feature {i}')
    plt.legend()
    plt.show()

"""
###############################################################################################################################################################################################################
##          Classification
###############################################################################################################################################################################################################
print("################################################################################################################################################################################################################################")
print("## Classification")
print("################################################################################################################################################################################################################################")

#Data creation
inputVector = dp.create_input('l1/')
length = len(inputVector[0])

y = []
X = []
print(len(inputVector))
print(len(inputVector[0]))
for j in range(len(inputVector)):
    for i in range(len(inputVector[j])):
        if (i+1) % length == 0:
            y.append(inputVector[j][i])

for i in range(len(inputVector)):
    dataPoint = []
    for j in range(len(inputVector[i])):
        if (j + 1) % length != 0:
            dataPoint.append(inputVector[i][j])
        if ((j + 1) % length == 0):
            X.append(dataPoint)
            dataPoint = []

print(np.array(y).shape)
yNp = np.array(y)
XNp = np.array(X)
np.savetxt('help.txt', yNp.T)
yPositive = yNp[yNp > 0]
xPositive = XNp[yNp > 0]
yNpNegative = yNp[yNp <= 0]
xNpNegative = XNp[yNp <= 0]
yNegative = sparsify(yNpNegative, len(xPositive))
xNegative = xNpNegative[np.random.choice(xNpNegative.shape[0], len(xPositive), replace=False)]
X = np.vstack((xNegative, xPositive))
y = np.hstack((yNegative, yPositive))

print("Y: ")
print(y)
print(X.shape)
# Scatter plot of features colored by class
for i in range(length-1):
    plt.scatter(range(len(y)), X[:, i], color=[["black", "red"][int(value)] for value in y], label='Data Points')
    plt.xlabel('Indices')
    plt.ylabel(f'Feature {i+1}')
    plt.title(f'Feature {i+1} Scatter Plot')
    plt.legend()
    plt.show()
print("Sum of y", sum(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(np.sum(X_train))
print(np.sum(X_test))
np.savetxt('array.txt', X)

"""
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
"""

#Pipeline

# Create the pipeline
pipeline = Pipeline([
    ('nystroem', Nystroem()),  # Nystroem transformer
    ('svc', LinearSVC(dual="auto"))  # Linear Support Vector Classifier
])


# Define the parameter grid for GridSearchCV
param_grid = {
    'nystroem__kernel': ['rbf'],
    'nystroem__gamma': [0.5, 0.75, 1.0],
    'nystroem__n_components': [10, 20, 30],
    'svc__C': [0.1, 1, 10],
    'svc__max_iter': [1000, 5000]
}



# Set up the GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model using GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)
"""
"""
svm = SVC()

# Define parameter grid to test different kernels, C values, and maximum iterations
param_grid = {
    'C': [0.1, 1, 10, 100],       # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Test different kernels
    'gamma': ['scale', 'auto']    # Gamma parameter for RBF, poly, and sigmoid kernels
}

print("Starting GridSearchCV for SVM with different kernels")

# Initialize GridSearchCV with SVC
grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', verbose=3, cv=5)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

print("GridSearchCV completed")
"""
# Get best parameters and model from grid search
"""
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

best_svm = grid_search.best_estimator_

# Predict on the test data
y_pred = best_svm.predict(X_test)
print(y_pred)

# Calculate accuracy and other classification metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
"""
#LinearSVM


svm = LinearSVC(dual="auto")

param_grid = {'C': [0.1, 1, 10, 100],
              'max_iter': [1000, 5000]}

print("Starting GridSearchCV for SVM")
# Initialize GridSearchCV
grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', verbose = 3, cv=5)
grid_search.fit(X_train, y_train)
print("GridSearchCV completed")
# Best parameters and model
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
best_svm = grid_search.best_estimator_

y_pred = best_svm.predict(X_test)
print(y_pred)

# Calculate accuracy and other classification metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

model_params = {
    "coef": best_svm.coef_.tolist(),
    "intercept": best_svm.intercept_.tolist(),
}

with open('linear_svc_params.json', 'w') as f:
    json.dump(model_params, f)
"""
joblib_file = "models/svm_linear_one_time_metric.pkl"
joblib.dump(best_svm, joblib_file)
"""
"""
#Analysis with adapted recall

y_proba = best_svm.predict_proba(X_test)[:, 1]

# Adjust the threshold
threshold = 0.03  # Example threshold
y_pred_adjusted = (y_proba >= threshold).astype(int)

# Calculate accuracy and other classification metrics with adjusted threshold
accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
class_report_adjusted = classification_report(y_test, y_pred_adjusted)

print(f"Accuracy with adjusted threshold: {accuracy_adjusted:.3f}")
print("Confusion Matrix with adjusted threshold:")
print(conf_matrix_adjusted)
print(conf_matrix_adjusted[0,1])
print("Classification Report with adjusted threshold:")
print(class_report_adjusted)

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
for i, txt in enumerate(thresholds):
    if i % 5 == 0:  # Annotate every 5th threshold for clarity
        plt.annotate(f'{txt:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(-10,-10), ha='center')
plt.show()
"""
"""
joblib_file = "models/svm_model_one_time_metric.pkl"
joblib.dump(best_svm, joblib_file)
"""
# Scatter plot of actual vs predicted values for features
for i in range(length-1):
    plt.scatter(X_test[:, i], y_test, color='black', label='Actual Data')
    plt.scatter(X_test[:, i], y_pred, color='red', label='Predicted Data')
    plt.xlabel(f'Feature {i+1}')
    plt.ylabel('Target')
    plt.title(f'SVM Classification: Actual vs Predicted (Feature {i+1})')
    plt.legend()
    plt.show()
    


###############################################################################################################################################################################################################
##          Pruning
###############################################################################################################################################################################################################
print("################################################################################################################################################################################################################################")
print("## Pruning")
print("################################################################################################################################################################################################################################")





#loaded_model = joblib.load("models/svr_model_one_time_metric.pkl")
#pruning.process_directory_and_predict_svr(loaded_model, "l1/", 80)
#pruning.process_directory_and_predict_svr(loaded_model, "l2/", 80)
"""



#loaded_model = joblib.load("models/svm_rbf.pkl")
pruning.process_directory_and_predict_svm(best_svm, "l2/")
