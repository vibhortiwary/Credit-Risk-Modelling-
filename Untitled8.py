#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[2]:


import pandas as pd

# Load the dataset
data = pd.read_csv('loan_detection.csv')


# In[3]:


# Check the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())


# In[4]:


# One-hot encoding for categorical features
data = pd.get_dummies(data, drop_first=True)


# In[5]:


X = data.drop('Loan_Status_label', axis=1)
y = data['Loan_Status_label']


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[14]:


import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = model.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[15]:


import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('loan_detection.csv')

# Data preprocessing 
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop('Loan_Status_label', axis=1)
y = data['Loan_Status_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[16]:


# Display the first few rows
print(data.head())

# Check the data types and for missing values
print(data.info())
print(data.isnull().sum())


# In[17]:


# Example: Fill missing values with the median for numerical columns
data.fillna(data.median(), inplace=True)

# Alternatively, drop rows with missing values
# data.dropna(inplace=True)


# In[19]:


# One-hot encoding for categorical features
data = pd.get_dummies(data, drop_first=True)


# In[22]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['age', 'campaign', 'pdays', 'previous']  # Example numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# In[28]:


# Example: Create a feature that indicates if the applicant has been contacted previously
data['contacted_before'] = (data['previous'] > 0).astype(int)

# Example: Create an age group feature
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
data = pd.get_dummies(data, columns=['age_group'], drop_first=True)


# In[29]:


# Drop features that are not useful for prediction
data.drop(['Loan_Status_label'], axis=1, inplace=True)  # Example of dropping the target variable from features


# In[30]:


# Define features and target variable
X = data.drop('Loan_Status_label', axis=1)  # Assuming 'Loan_Status_label' is the target variable
y = data['Loan_Status_label']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




