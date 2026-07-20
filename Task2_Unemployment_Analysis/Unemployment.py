#Task 2
#UNEMPLOYMENT ANALYSIS USING PYTHON

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment in India.csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nCOLUMN NAMES")
print(df.columns)

print("\nSHAPE OF DATASET")
print(df.shape)

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())


# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove missing values
df = df.dropna()

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

print("\nSHAPE AFTER CLEANING")
print(df.shape)


# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

trend = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10, 5))
plt.plot(trend.index, trend.values, marker="o")

plt.title("Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

state_unemployment = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(12, 6))
state_unemployment.sort_values().plot(kind="bar", color="skyblue")

plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


df["Year"] = df["Date"].dt.year

yearly_unemployment = df.groupby("Year")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(8, 5))
yearly_unemployment.plot(kind="bar", color="orange")

plt.title("Average Unemployment Rate by Year")
plt.xlabel("Year")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

df["Month"] = df["Date"].dt.month_name()

monthly_unemployment = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

monthly_unemployment = monthly_unemployment.reindex(months)

plt.figure(figsize=(10, 5))
monthly_unemployment.plot(kind="line", marker="o", color="green")

plt.title("Average Monthly Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.grid(True)
plt.tight_layout()
plt.show()

print("\nProject Completed Successfully!")
