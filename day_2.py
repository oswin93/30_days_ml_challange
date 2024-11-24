# this problem is classification and we will implement logistic regression

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# step 1 loading the dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data,columns= iris_data.feature_names)
iris_df['species']=pd.Categorical.from_codes(iris_data.target,iris_data.target_names)

# step 2 prepare the data separate features and target

X= iris_df.iloc[:,:-1] # all rows except last column
y= iris_df.iloc[:,-1] # last colum is the target

# step 3 splitting the dataset into train and validation process
# we will split 80% for training and 20% for testing purpose so test size =0.2
# we want it to generate split with same random ness on every run hence setting base random state =42
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2,random_state=42)

# step 4 train the model
# we try to achive minimum loss function hence incrrease the max iteration count to 200
# default is 100
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# step 5 make predictions
predictions = model.predict(X_test)

# step 6 evaluate the model
# propotion of correctly predicted observations to the total observations
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("accuracy is :", accuracy)
print("confusion matrix:",conf_matrix)
print("classification report", class_report)


# the model predicts correctly for all the test data
# lets change the max_iter to 100 and see what happens
# same result even with 100 iterations

# step 7 Visualization

plt.figure(figsize=(10,7)) #window size
sns.heatmap(conf_matrix, annot= True, fmt="d", xticklabels=iris_data.target_names, yticklabels=iris_data.target_names)
plt.title("confusion matrix")
plt.xlabel("predicted species")
plt.ylabel("actual species")
plt.show()