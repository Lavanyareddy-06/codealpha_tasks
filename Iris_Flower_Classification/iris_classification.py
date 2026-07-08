
# IRIS FLOWER CLASSIFICATION
# CODEALPHA TASK 3
# -------------------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Iris.csv")

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



# Remove Id column
df = df.drop("Id", axis=1)

# Input Features
X = df.drop("Species", axis=1)

# Output Label
y = df["Species"]

print("\nINPUT FEATURES")
print(X.head())

print("\nOUTPUT LABEL")
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(accuracy)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# : PREDICT NEW FLOWER
# -------------------------

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(new_flower)

print("\nNEW FLOWER PREDICTION")
print(prediction[0])
