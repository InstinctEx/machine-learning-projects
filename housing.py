#https://www.kaggle.com/datasets/ashydv/housing-dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Housing.csv')

X = data[['bedrooms', 'area']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y)

def menu():
    bedrooms = float(input("Enter number of bedrooms: "))
    area = float(input("Enter area: "))
    input_df = pd.DataFrame([[bedrooms, area]], columns=['bedrooms', 'area']) 
    predicted_price = linr.predict(input_df)
    print(f"Predicted price: {predicted_price[0]}")
    predicted_category = dtcr.predict(input_df)
    print(f"Category: {predicted_category[0]}")

def stats(X, Y):
    predictions = linr.predict(X)
    print(f"Mean Price: {np.mean(Y)}")
    print(f"Mean Absolute Error: {mean_absolute_error(Y, predictions)}")
    print(f"Median: {np.median(Y)}")

linr = LinearRegression()
linr.fit(X_train, y_train)

def category(price):
    if price < 2000000:
        return "Φθηνο"
    elif price < 5000000:
        return "Ενδιαμεσο"
    else:
        return "Ακριβο"

data['categories'] = data['price'].apply(category)
y_cat = data['categories']

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_cat)

dtcr = DecisionTreeClassifier()
dtcr.fit(X_train_cat, y_train_cat)

menu()
stats(X_test, y_test)
