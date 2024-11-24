# recognize handwritten digits with k-nearest neighbours on mnist
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# step1 load the mnist data
mnist_data = fetch_openml("mnist_784",version=1, as_frame=True, parser="auto")

# the image of size 28 x 28 pixel is already flattened into 784 feature vector

# step2 : preprocess the data
X = mnist_data.data
y= mnist_data.target

# step 3 Normalize the data . already the data is in 0,1 format
# it is better to normalize the data again as KNeighborsClassifier works on that
X/= 255.0

# step -4 splitting training data and test data in the ration 80 and 20

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

# step 5 create and train the model
# we start with k=3 which is normal and initial choice
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# step 6 making predictions and evaluate the model
predictions = model.predict(X_test)

# Evaluate actual value against predicted values
accuracy = accuracy_score(y_test, predictions)
# you can see the true positives in diagonally and all other FP,FN, TN in other columns
# the TP are in good number that is why the accuracy is 0.97
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("accuracy is :", accuracy)
print("confusion matrix:",conf_matrix)
print("classification report", class_report)

# step 7 Visualization

plt.figure(figsize=(10,7)) #window size
sns.heatmap(conf_matrix, annot= True, fmt="g", cmap="Blues", xticklabels=mnist_data.target_names, yticklabels=mnist_data.target_names)
plt.title("confusion matrix")
plt.xlabel("predicted values")
plt.ylabel("actual values")
plt.show()

# when K=1 accuracy improved
# when k=5 accuracy is 0.970
# when k=10 accuracy decreased to 0.965

# at k=3 accuracy is best 0.9712
