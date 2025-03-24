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
import pandas as pd
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

def extract_features_labels(input_vector):
    length = len(input_vector[0])
    y = []
    X = []
    for j in range(len(input_vector)):
        for i in range(len(input_vector[j])):
            if (i + 1) % length == 0:
                y.append(input_vector[j][i])

    for i in range(len(input_vector)):
        dataPoint = []
        for j in range(len(input_vector[i])):
            if (j + 1) % length != 0:
                dataPoint.append(input_vector[i][j])
            if ((j + 1) % length == 0):
                X.append(dataPoint)
                dataPoint = [] # The last element is the label.
    return np.array(X), np.array(y)


# Helper function to balance the dataset based on positive examples.
def balance_dataset(X, y):
    yPositive = y[y > 0]
    xPositive = X[y > 0]
    yNpNegative = y[y <= 0]
    xNpNegative = X[y <= 0]
    yNegative = sparsify(yNpNegative, len(xPositive))
    #sparsify does not work here because xNpNegative is an numpy array with both multiple columns and rows.
    xNegative = xNpNegative[np.random.choice(xNpNegative.shape[0], len(xPositive), replace=False)]
    #put the positive and negative data points back together
    X_balanced = np.vstack((xNegative, xPositive))
    y_balanced = np.hstack((yNegative, yPositive))
    return X_balanced, y_balanced


def save_features_to_csv(X_train, y_train, X_test, y_test):
    # Process training data
    y_train = y_train.reshape(-1, 1)
    train_data = np.hstack((X_train, y_train))
    df_train = pd.DataFrame(train_data)
    df_train.to_csv('features_l1_best_random.csv', index=False, header=False)

    # Process test data
    y_test = y_test.reshape(-1, 1)
    test_data = np.hstack((X_test, y_test))
    df_test = pd.DataFrame(test_data)
    df_test.to_csv('features_l1_best.csv', index=False, header=False)

def create_data():
    # Data creation
    # === Training data creation ===
    inputVector = dp.create_input('l1Random/', "tightenedWindowsRandom/", "infeasibleArcsRandom/", "outRandom/")
    length = len(inputVector[0])
    X_train, y_train = extract_features_labels(inputVector)
    X_train, y_train = balance_dataset(X_train, y_train)

    # === Test data creation ===
    inputVector_test = dp.create_input('l1/', "tightenedWindows/", "infeasibleArcs/", "out/")
    X_test, y_test = extract_features_labels(inputVector_test)
    X_test, y_test = balance_dataset(X_test, y_test)

    # === Scale the data ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #saves the features and labels into csv files.
    #save_features_to_csv(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


###############################################################################################################################################################################################################
##          Classification
###############################################################################################################################################################################################################
print("################################################################################################################################################################################################################################")
print("## Classification")
print("################################################################################################################################################################################################################################")



def svm_nystroem ():
    X_train, y_train, X_test, y_test = create_data()
    # === Pipeline ===
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

    # Get best parameters and model from grid search
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    best_svm = grid_search.best_estimator_

    # Predict on the test data
    y_pred = best_svm.predict(X_test)

    # Calculate accuracy and other classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)


def svm_linear ():
    X_train, y_train, X_test, y_test = create_data()
    #LinearSVM
    svm = LinearSVC(dual="auto")
    param_grid = {'C': [0.1, 1, 10, 100],
                  'max_iter': [1000, 5000]}

    # Initialize GridSearchCV
    grid_search = GridSearchCV(svm, param_grid, scoring='recall', verbose = 3, cv=5)
    grid_search.fit(X_train, y_train)
    print("GridSearchCV completed")
    # Best parameters and model
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)

    # Calculate accuracy and other classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    #save the best model for later use
    model_params = {
        "coef": best_svm.coef_.tolist(),
        "intercept": best_svm.intercept_.tolist(),
    }
    with open('linear_svc_params.json', 'w') as f:
        json.dump(model_params, f)



svm_nystroem()
#svm_linear()



###############################################################################################################################################################################################################
##          OBSOLETE AND OUTDATED FROM HERE. JUST LEFT IN FOR COMPLETNESS.
###############################################################################################################################################################################################################

###############################################################################################################################################################################################################
##          Regression !!!OUTDATED AND NOT USED ANYMORE!!!
###############################################################################################################################################################################################################

"""
print("################################################################################################################################################################################################################################")
print("## Regression")
print("################################################################################################################################################################################################################################")


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
##          Pruning !!!OBSOLETE, THIS IS NOW DONE IN JULIA CODE!!!
###############################################################################################################################################################################################################
"""
print("################################################################################################################################################################################################################################")
print("## Pruning")
print("################################################################################################################################################################################################################################")





#loaded_model = joblib.load("models/svr_model_one_time_metric.pkl")
#pruning.process_directory_and_predict_svr(loaded_model, "l1/", 80)
#pruning.process_directory_and_predict_svr(loaded_model, "l2/", 80)




#loaded_model = joblib.load("models/svm_rbf.pkl")
#pruning.process_directory_and_predict_svm(best_svm, "l2/")
"""
