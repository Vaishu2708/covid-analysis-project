#!/usr/bin/env python
# coding: utf-8

# In[32]:


# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[33]:


# Style for better visuals
sns.set(style="whitegrid")

# 2. LOAD DATASET
df = pd.read_csv("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/data/patient.csv")
print("\n Load Dataset...\n")


# In[34]:


# 3. DATA PREPROCESSING

print("\n DATA OVERVIEW")
print(df.head())

print("\n DATA INFORMATION")
print(df.info())

# Convert dates
df['confirmed_date'] = pd.to_datetime(df['confirmed_date'], errors='coerce')
df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')

# Calculate recovery days
df['recovery_days'] = (df['released_date'] - df['confirmed_date']).dt.days

# Clean data
df = df[df['recovery_days'] >= 0]
df['contact_number'] = df['contact_number'].fillna(0)

# Age calculation
df['age'] = 2020 - df['birth_year']

print("\n Data Preprocessing Completed")


# In[35]:


# 4 DESCRIPTIVE STATISTICS

print("\n📌 Descriptive Statistics Summary:\n")

print(df[['age', 'contact_number', 'infection_order', 'recovery_days']].describe())

print("\nAverage Age:", df['age'].mean())
print("Average Recovery Days:", df['recovery_days'].mean())


# In[36]:


# 5. EXPLORATORY DATA ANALYSIS

print("\n EXPLORATORY DATA ANALYSIS\n")

# Gender Distribution

plt.figure()
sns.countplot(x='sex', data=df)
plt.title(" Gender Distribution of COVID-19 Patients", fontsize=14, fontweight='bold')
plt.xlabel("Gender")
plt.ylabel("Number of Cases")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/gender distribution.png")
plt.show()

# Age Distribution

plt.figure()
sns.histplot(df['age'], bins=20, kde=True)
plt.title(" Age Distribution of Patients", fontsize=14, fontweight='bold')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/age distribution.png")
plt.show()

# Region-wise Cases

plt.figure(figsize=(10,5))
df['region'].value_counts().head(10).plot(kind='bar')
plt.title(" Top 10 Regions with Highest Cases", fontsize=14, fontweight='bold')
plt.xlabel("Region")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/Regions.png")
plt.show()

# Infection Sources

plt.figure(figsize=(10,5))
df['infection_reason'].value_counts().head(10).plot(kind='bar')
plt.title(" Major Infection Sources", fontsize=14, fontweight='bold')
plt.xlabel("Infection Reason")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/Infection.png")
plt.show()


# In[37]:


# 6. RECOVERY ANALYSIS

print("\n RECOVERY ANALYSIS\n")

plt.figure()
sns.histplot(df['recovery_days'], bins=20, kde=True)
plt.title(" Recovery Time Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Recovery Days")
plt.ylabel("Frequency")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/Recovery.png")
plt.show()

avg_recovery = df['recovery_days'].mean()
print(f" Average Recovery Time: {avg_recovery:.2f} days")


# In[38]:


# 7. CORRELATION ANALYSIS

print("\n CORRELATION ANALYSIS\n")

plt.figure()
sns.heatmap(
    df[['age','contact_number','infection_order','recovery_days']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix of Key Features", fontsize=14, fontweight='bold')
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/correlation.png")
plt.show()


# In[39]:


# 8. LINEAR REGRESSION MODEL (HIGHLIGHTED)

print(" LINEAR REGRESSION MODEL - RECOVERY PREDICTION ")

# Features & Target
X = df[['age', 'contact_number', 'infection_order']].fillna(0)
y = df['recovery_days']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Training Model...")

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

print(" Model Training Completed")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
print(f"\n Model Performance (R² Score): {r2:.4f}")

# Coefficients Table
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n Feature Importance (Coefficients Table):")
print(coeff_df.to_string(index=False))

# Actual vs Predicted Graph

plt.figure()
plt.scatter(y_test, y_pred)
plt.title(" Actual vs Predicted Recovery Days", fontsize=14, fontweight='bold')
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/predicted.png")
plt.xlabel("Actual Recovery Days")
plt.ylabel("Predicted Recovery Days")
plt.show()

print("\n Linear Regression Analysis Completed")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




