# ---------------------------------------
# CAR PRICE PREDICTION USING MACHINE LEARNING
# ---------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error



df = pd.read_csv("car data.csv")

print("FIRST 5 ROWS")
print(df.head())


print("\nDATASET INFORMATION")
print(df.info())

print("\nCOLUMN NAMES")
print(df.columns)

print("\nSHAPE OF DATASET")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())



label = LabelEncoder()

df["Fuel_Type"] = label.fit_transform(df["Fuel_Type"])
df["Selling_type"] = label.fit_transform(df["Selling_type"])
df["Transmission"] = label.fit_transform(df["Transmission"])

# Remove Car_Name column
df = df.drop("Car_Name", axis=1)
-

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nINPUT FEATURES")
print(X.head())

print("\nOUTPUT LABEL")
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)



model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")



prediction = model.predict(X_test)

print("\nR2 SCORE")
print(r2_score(y_test, prediction))

print("\nMEAN ABSOLUTE ERROR")
print(mean_absolute_error(y_test, prediction))

# ---------------------------------------
#  NEW CAR PRICE PREDICTION
# ---------------------------------------

sample = pd.DataFrame(
    [[2018, 5.5, 35000, 1, 1, 1, 0]],
    columns=X.columns
)

predicted_price = model.predict(sample)

print("\nPREDICTED CAR PRICE")
print("Estimated Selling Price:", round(predicted_price[0], 2), "Lakhs")
