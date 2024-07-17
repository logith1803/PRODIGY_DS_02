#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_data = pd.read_csv(url)

# Display the first few rows of the dataset
print(titanic_data.head())
# Check for missing values
print(titanic_data.isnull().sum())

# Handle missing values for 'Age' and 'Fare'
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].mode()[0], inplace=True)

# Verify if missing values are handled
print(titanic_data.isnull().sum())

# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=titanic_data, palette='Set1')
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=titanic_data, palette='Set2')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Distribution of age
plt.figure(figsize=(8, 6))
sns.histplot(titanic_data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Survival rate by age
plt.figure(figsize=(10, 6))
sns.histplot(x='Age', hue='Survived', data=titanic_data, bins=20, kde=True, palette='Set1')
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()


# In[ ]:




