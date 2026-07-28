import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# making my own small dataset since i dont have a csv rn
data = {
    "Job Title": ["Data Analyst", "Data Scientist", "Software Engineer", "Data Analyst",
                  "Manager", "Software Engineer", "Data Scientist", "Manager",
                  "Data Analyst", "Software Engineer", "Data Scientist", "Manager",
                  "Data Analyst", "Software Engineer", "Data Scientist"],
    "Years of Experience": [1, 3, 2, 4, 8, 5, 6, 10, 2, 7, 9, 12, 3, 4, 1],
    "Salary": [35000, 65000, 55000, 45000, 95000, 75000, 85000, 110000,
               40000, 82000, 98000, 130000, 42000, 68000, 60000]
}

df = pd.DataFrame(data)

print(df)

# checking if there is any missing data
print(df.isnull().sum())

# just dropping duplicates just in case
df = df.drop_duplicates()

# job title is text so model wont understand it, converting to numbers
le = LabelEncoder()
df["Job Title"] = le.fit_transform(df["Job Title"])

# x is the input, y is what we want to predict
x = df[["Years of Experience", "Job Title"]]
y = df["Salary"]

# splitting data, 80 percent train 20 percent test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# creating the model
model = LinearRegression()

# training it
model.fit(x_train, y_train)

# now predicting salary for test data
y_pred = model.predict(x_test)

print("Predicted:", y_pred)
print("Actual:", y_test.values)

# checking how good the model is
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE =", mse)
print("R2 score =", r2)

# plotting actual vs predicted to see how close they are
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")  # this is just the perfect line
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()