import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

canada = pd.read_csv('canada_house_listings.csv',encoding = 'ISO-8859-1')
canada.head()

# location data is not accurate for the individual property but they are for the relevant city.
# therefore let's remove unwanted columns such as median family income and location data  population
canada = canada.drop(['Address','Province','Population','Latitude','Longitude','Median_Family_Income'], axis=1)
canada.head()

#let's check for null values and data types of the dataset
print(canada.isnull().sum())
print(canada.info())

#describe the dataset
canada.describe()

#let's get dummies for the city variable
canada = pd.get_dummies(canada, columns=['City'], drop_first=True)
canada.head()

#split the dataset into x and y
x_can = canada.drop(['Price'], axis=1)
y_can = canada['Price']

#splitting into testing and traning models
x_can_train, x_can_test, y_can_train, y_can_test = train_test_split(x_can, y_can, test_size=0.2, random_state=42)

#fitting the model
model = LinearRegression()
model.fit(x_can_train, y_can_train)

#predicting the prices
y_can_pred = model.predict(x_can_test)

#Evaluate the Model (Error Metrics)
mae = mean_absolute_error(y_can_test, y_can_pred)
mse = mean_squared_error(y_can_test, y_can_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

#Visualization of predictions vs actual prices
plt.scatter(y_can_test, y_can_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices")
plt.show()
