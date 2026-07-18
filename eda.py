# ============================
# Exploratory Data Analysis (EDA)
# Titanic Dataset
# ============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
print(os.listdir())
print(os.listdir("data"))

# Display Settings
pd.set_option('display.max_columns', None)

# Load Dataset
df = pd.read_csv("data/titanic.csv")

print("First 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Values")
print(df.duplicated().sum())

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Fill Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna("Unknown", inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# -----------------------------------
# Data Visualization
# -----------------------------------

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Countplot
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title("Male vs Female")
plt.show()

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'])
plt.title("Age Boxplot")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title("Age vs Fare")
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
df['Survived'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['red','green']
)
plt.title("Survival Percentage")
plt.ylabel("")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,7))
numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df[['Age','Fare','Pclass','Survived']])
plt.show()

# -----------------------------------
# Insights
# -----------------------------------

print("\n========== PROJECT INSIGHTS ==========")

print("1. Dataset contains", df.shape[0], "rows.")

print("2. Male and Female passenger distribution analyzed.")

print("3. Survival distribution analyzed.")

print("4. Passenger class distribution analyzed.")

print("5. Missing values handled.")

print("6. Correlation between numerical columns calculated.")

print("7. Outliers identified using Boxplot.")

print("8. Relationship between Age and Fare analyzed.")

print("\nEDA Completed Successfully.")