# diagnose breast cancer as malignant or benign using a decision tree
# 3rd classification problem in the series
from unittest import SkipTest

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# step1 load the datset
data = load_breast_cancer()

# create pandas dataframe
# the data.frame creates none
# step 2 create features and target set
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# step 3 splitting the datset into training and testing
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)


# step -4 create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# step 5 Make predictions and Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("accuracy is :", accuracy)
print("confusion matrix:",conf_matrix)
print("classification report", class_report)


# a confusion matrix shows the number of:
# true positives : correctly predicted malignant cases
# true negatives : correctly predicted benign cases
# false positives : benign cases incorrectly predicted as malignant (type I error)
# false negatives : malignant cases incorrectly predicted as benign (also known as Type II error

# step 6 Visualization
plt.figure(figsize=(10,7)) #window size
sns.heatmap(conf_matrix, annot= True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("confusion matrix")
plt.xlabel("Predicted ")
plt.ylabel("Actual ")
plt.show()

# true positives : correctly predicted malignant cases are 40
# true negatives : correctly predicted benign cases are 68

plt.figure(figsize=(10,7))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

