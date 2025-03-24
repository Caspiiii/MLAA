import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import dataprocessing as dp

########################################################################################################################
##
##                      -- KNN FILE --
##  Train and analyse a K-nearest neighbor to compare with the support vector machines
##
########################################################################################################################



## Deal with data being heavily inbalanced (a lot more negative)
def sparsify(input, size):
    """Sparsify the input down to the size given in size"""
    num_to_select = size
    selected_elements = np.random.choice(input, num_to_select, replace=False)
    return selected_elements

# If the feature value is <= threshold -> class 1 else class 0
class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=1):
        self.threshold = threshold

    def fit(self, X, y=None):
        # No fitting is necessary
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # Convert input to a numpy array and flatten it.
        X = np.array(X).flatten()
        # Return 1 if the neighbor order is <= threshold, else 0.
        return (X <= self.threshold).astype(int)


#Data creation

inputVector = dp.create_input_knn('l1Random/', "infeasibleArcsRandom/","outRandom/")
length = len(inputVector[0])

y = []
X = []
for i in range(len(inputVector)):
    for j in range(len(inputVector[i])):
        if (j+1) % length == 0:
            y.append(inputVector[i][j])

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
yPositive = yNp[yNp > 0]
xPositive = XNp[yNp > 0]
yNpNegative = yNp[yNp <= 0]
xNpNegative = XNp[yNp <= 0]
yNegative = sparsify(yNpNegative, len(xPositive))
xNegative = xNpNegative[np.random.choice(xNpNegative.shape[0], len(xPositive), replace=False)]
X_train = np.vstack((xNegative, xPositive))
y_train = np.hstack((yNegative, yPositive))

#test data creation
inputVector_test = dp.create_input_knn('l1/', "infeasibleArcs/","out/")
length = len(inputVector_test[0])

y_test = []
X_test = []
print(len(inputVector_test))
print(len(inputVector_test[0]))
for i in range(len(inputVector_test)):
    for j in range(len(inputVector_test[i])):
        if (j+1) % length == 0:
            y_test.append(inputVector_test[i][j])

for i in range(len(inputVector_test)):
    dataPoint = []
    for j in range(len(inputVector_test[i])):
        if (j + 1) % length != 0:
            dataPoint.append(inputVector_test[i][j])
        if ((j + 1) % length == 0):
            X_test.append(dataPoint)
            dataPoint = []

print(np.array(y_test).shape)
y_test_Np = np.array(y_test)
X_test_Np = np.array(X_test)
y_test_Positive = y_test_Np[y_test_Np > 0]
X_test_Positive = X_test_Np[y_test_Np > 0]
y_test_NpNegative = y_test_Np[y_test_Np <= 0]
X_test_NpNegative = X_test_Np[y_test_Np <= 0]
y_test_Negative = sparsify(y_test_NpNegative, len(X_test_Positive))
x_test_Negative = X_test_NpNegative[np.random.choice(X_test_NpNegative.shape[0], len(X_test_Positive), replace=False)]
X_test = np.vstack((x_test_Negative, X_test_Positive))
y_test = np.hstack((y_test_Negative, y_test_Positive))


# search for the best threshold over a range of possible neighbor ranks.
min_threshold = int(np.min(X_train))
max_threshold = int(np.max(X_train))
possible_thresholds = range(min_threshold, max_threshold + 1)

# Use StratifiedKFold to ensure the class proportions are preserved in each fold.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

threshold_results = {}

for threshold in possible_thresholds:
    clf = ThresholdClassifier(threshold=threshold)
    # Use cross_val_score to compute the accuracy for the current threshold
    scores = cross_val_score(clf, X_train.reshape(-1, 1), y_train, cv=skf, scoring='accuracy')
    threshold_results[threshold] = np.mean(scores)
    print(f"Threshold {threshold}: CV Accuracy = {np.mean(scores):.3f}")

# Select the threshold that gives the highest cross-validation accuracy.
best_threshold = max(threshold_results, key=threshold_results.get)
print("\nBest threshold (k):", best_threshold)
print("Best relative value for k based on n:", best_threshold/max_threshold)
print("Cross-validated Accuracy:", threshold_results[best_threshold])

# Train the classifier on the full training set using the best threshold.
best_clf = ThresholdClassifier(threshold=best_threshold)
best_clf.fit(X_train.reshape(-1, 1), y_train)

# Evaluate on the test set.
y_pred = best_clf.predict(X_test.reshape(-1, 1))
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", test_accuracy)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))