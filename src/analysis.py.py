#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


# Load dataset
df = pd.read_csv("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/data/patient.csv")

# Convert date columns
df['confirmed_date'] = pd.to_datetime(df['confirmed_date'], errors='coerce')
df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')

# Feature Engineering
df['age'] = 2020 - df['birth_year']
df['recovery_days'] = (df['released_date'] - df['confirmed_date']).dt.days

# Clean data
df = df.dropna(subset=['age', 'recovery_days'])


# In[17]:


# Gender distribution
plt.figure()
sns.countplot(x='sex', data=df)
plt.title("Gender Distribution")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/gender.png")

# Age distribution
plt.figure()
sns.histplot(df['age'], bins=20)
plt.title("Age Distribution")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/age.png")

# Region-wise cases
plt.figure()
df['region'].value_counts().head(10).plot(kind='bar')
plt.title("Top Regions")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/region.png")

# Infection reasons
plt.figure()
sns.countplot(y='infection_reason', data=df)
plt.title("Infection Sources")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/infection.png")


# In[19]:


corr = df[['age', 'contact_number', 'infection_order', 'recovery_days']].corr()

plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.savefig("C:/Users/Vaishnavi/OneDrive/Desktop/covid-analysis-project/outputs/correlation.png")


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['age', 'contact_number', 'infection_order']].fillna(0)
y = df['recovery_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[21]:


model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("Model R2 Score:", score)


# In[ ]:




