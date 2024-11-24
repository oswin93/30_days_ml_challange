# predict house price using simple linear regression
# data is from sklearn library
from statistics import linear_regression

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# step 1 load the dataset
housing_data= fetch_california_housing(as_frame=True)
housing_data_df = housing_data.frame

# Access the data and target
X=housing_data_df[['MedInc']]
y = housing_data_df['MedHouseVal']

# step 2 split the data 80% for training and 20% for validating
X_train, X_test, y_train, y_test =train_test_split(X , y, test_size=0.2, random_state=42)


# step 3 create and train the model
model =LinearRegression()
model.fit(X_train, y_train)

# step 4 Evaluation

predictions =model.predict(X_test)
mse=mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
#lower the rmse, the better your model has performed in predicting the target variable
print("the root mean squared error is :", rmse)


# step 5 Visualization
plt.scatter(X_test, y_test, label='True value', alpha=0.5)
plt.plot(X_test, predictions,label='Predicted value', color ='red')
plt.xlabel('MedInc')
plt.ylabel('median house value')
plt.legend()
plt.show()
