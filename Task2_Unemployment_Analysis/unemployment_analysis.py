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

df.columns = df.columns.str.strip()

df = df.dropna()

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

print("\nSHAPE AFTER CLEANING")
print(df.shape)



import matplotlib.pyplot as plt
import seaborn as sns


df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)


trend = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

# Plot line graph
plt.figure(figsize=(10,5))
plt.plot(trend.index, trend.values, marker='o')

plt.title("Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()


# Calculate average unemployment rate for each state
state_unemployment = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()

# Create bar chart
plt.figure(figsize=(12,6))
state_unemployment.sort_values().plot(kind="bar", color="skyblue")

plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()


# Create Year and Month columns
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month_name()

# Average unemployment rate by year
yearly_unemployment = df.groupby("Year")["Estimated Unemployment Rate (%)"].mean()

# Plot graph
plt.figure(figsize=(8,5))
yearly_unemployment.plot(kind="bar", color="orange")

plt.title("Average Unemployment Rate by Year")
plt.xlabel("Year")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()

# Calculate average unemployment rate by month
monthly_unemployment = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()

# Arrange months in correct order
months = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

monthly_unemployment = monthly_unemployment.reindex(months)

# Plot graph
plt.figure(figsize=(10,5))
monthly_unemployment.plot(kind="line", marker="o", color="green")

plt.title("Average Monthly Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.grid(True)

plt.tight_layout()

plt.show()
