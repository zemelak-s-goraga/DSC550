#!/usr/bin/env python
# coding: utf-8

# Data Mining (DSC550-T301_2245_1)
# 
# Assignement Week 9;
# 
# Author: Zemelak Goraga;
# 
# Date: 05/11/2024

# In[3]:


# Step 1: Import necessary libraries and load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[13]:


# Load the dataset
df = pd.read_csv('Loan_Train.csv')
print("Data loaded successfully.")


# In[14]:


# Step 2.1: Display first few rows of the dataset
print(df.head())


# In[15]:


# Step 2.2: Display last few rows of the dataset
print(df.tail())


# In[16]:


# Step 2.2.1: Display dataset information (datatypes and non-null values)
print(df.info())


# In[17]:


# Step 2.3: Drop the column “Loan_ID”
df.drop('Loan_ID', axis=1, inplace=True)
df.head()


# In[19]:


# Step 2.4: Drop any rows with missing data
df.dropna(inplace=True)
df


# In[20]:


# Step 2.5: Convert the categorical features into dummy variables
df = pd.get_dummies(df, drop_first=True)


# In[21]:


# Step 2.5.1: Confirm the above step
print(df.info())


# In[22]:


# Step 2.6: Split the data into training and test sets
X = df.drop('Loan_Status_Y', axis=1)
y = df['Loan_Status_Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


# Step 2.7: Create a pipeline with a min-max scaler and a KNN classifier
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier())
])


# In[24]:


# Step 2.8: Fit a default KNN classifier to the data with this pipeline
pipeline.fit(X_train, y_train)


# In[25]:


# Step 2.9: Report the model accuracy on the test set
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.2f}")


# In[27]:


# Step 2.10: Create a search space for the KNN classifier
param_grid = {'knn__n_neighbors': list(range(1, 11))}


# In[28]:


# Step 2.11: Fit a grid search with the pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[29]:


# Step 2.12: Find the accuracy of the grid search best model on the test set
best_knn_accuracy = grid_search.score(X_test, y_test)
print(f"Best KNN Model Accuracy on test set: {best_knn_accuracy:.2f}")


# In[32]:


# Import necessary libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Setup the pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', KNeighborsClassifier())
])

# Expanding the search space to include other models with correct parameter naming
param_grid = [
    {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': list(range(1, 11))},
    {'classifier': [LogisticRegression()], 'classifier__C': [0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier()], 'classifier__n_estimators': [10, 50, 100, 200]}
]

# Configure and run GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)  # Ensure you have X_train and y_train defined

# Output the best parameters and the corresponding score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[33]:


# Step 2.14: Best model and hyperparameters found
print("Best model and hyperparameters:", grid_search.best_params_)


# In[34]:


# Step 2.15: Find the accuracy of this model on the test set
best_model_accuracy = grid_search.score(X_test, y_test)
print(f"Best Model Accuracy on test set: {best_model_accuracy:.2f}")


# Step 2.16: Summarize the results:
# 
# The baseline accuracy of the KNN classifier was 0.78.
# The best accuracy achieved by tuning the KNN model was 0.79.
# The best overall model was a Logistic Regression with C=10, which not only had the highest cross-validation score of 0.81 but also demonstrated the best accuracy on the test set, 0.82.
# 

# In[ ]:





# Short Report:
# 
# Title: Optimization of Loan Approval Predictions Using Machine Learning Techniques
# 
# 
# Summary:
# 
# This report details the outcomes of an analysis focused on utilizing various machine learning models to predict loan approval outcomes. The analysis leveraged a dataset available on Kaggle, and the study involved comparing models such as K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest. The primary objective was to determine which model and its hyperparameters could most effectively predict loan approvals, thereby supporting financial institutions in their decision-making processes.
# 
# The project consisted of several key phases: data preparation, model training, extensive hyperparameter tuning via Grid Search CV, and evaluation based on model accuracy. The initial results from the baseline model using K-Nearest Neighbors showed an accuracy of 0.78 on the test set. Subsequent tuning improved KNN's accuracy slightly to 0.79. However, further exploration and tuning revealed that Logistic Regression, particularly with a regularization strength C of 10, outperformed other models, achieving the highest cross-validation score of 0.81 and an accuracy of 0.82 on the test set.
# 
# These findings indicate the superior predictive power of the Logistic Regression model after appropriate tuning, suggesting its potential utility in operational settings for enhancing loan decision processes. The insights garnered from this analysis recommend Logistic Regression as a robust tool for financial institutions aiming to optimize their loan approval predictions.
# 
# 
# Introduction:
# 
# The process of approving or rejecting loan applications is critical for financial institutions, impacting both business profitability and customer satisfaction. Automating this decision-making process using machine learning can significantly enhance both the speed and accuracy of loan approvals. The current project involves developing a predictive model using the 'Loan Approval Data Set' from Kaggle, which includes various applicant features. The objective is to predict whether a loan should be approved or not, based on historical data, thus reducing human error and bias in loan processing.
# 
# Statement of the Problem:
# 
# Loan approval processes are traditionally reliant on manual scrutiny of applicant details, which can be prone to errors and biases. This manual process is often time-consuming and inconsistent. With the increasing volume of loan applications, financial institutions face significant challenges in maintaining efficiency and decision accuracy. The need for a robust, scalable, and accurate decision-making tool is evident to handle the growing demands effectively. The problem addressed by this project is to evaluate and optimize different machine learning models to find the most reliable automated solution for predicting loan approvals.
# 
# Methodology:
# 
# The methodology involved several key steps: First, the dataset was downloaded from Kaggle and loaded into a Jupyter Notebook environment for analysis. Initial data inspection and preprocessing included handling missing values, encoding categorical variables, and dropping irrelevant features. The data was then split into training and testing sets.
# 
# For the modeling, a pipeline was constructed integrating a MinMax scaler and initially a KNN classifier. The model was evaluated using default settings before tuning hyperparameters using Grid Search with cross-validation. This process was expanded to include Logistic Regression and Random Forest classifiers to explore their performance under various configurations.
# 
# Model performance was primarily evaluated using accuracy as the metric. The Grid Search method facilitated the identification of the best hyperparameters for each model, optimizing prediction outcomes.
# 
# Results:
# 
# The results obtained from the analysis are:
# 
# Baseline KNN Model Accuracy: This was initially obtained using a simple pipeline with just a MinMaxScaler and a KNeighborsClassifier without any hyperparameter tuning.
# Accuracy on the test set: 0.78
# Best KNN Model After Grid Search: This result came from using a grid search specifically tuned for the number of neighbors in the KNN classifier.
# Best KNN Model Accuracy on the test set: 0.79
# Expanding the Model Search with Different Classifiers: This involved setting up a more complex grid search to evaluate different models including KNeighborsClassifier, LogisticRegression, and RandomForestClassifier with specific hyperparameters for each.
# Best parameters found: Logistic Regression with C=10
# Best cross-validation score: 0.81
# Best Overall Model Performance on Test Set: The accuracy of the best model (Logistic Regression with C=10) when evaluated on the test set.
# Best Model Accuracy on the test set: 0.82
# 
# 
# Discussion:
# 
# In this analysis, the effectiveness of different classifiers in predicting loan approvals were eveluated using a dataset initially consisting of 614 entries. The preliminary analysis focused on a baseline KNeighborsClassifier integrated within a pipeline that included MinMaxScaler normalization. This baseline model achieved an accuracy of 0.78 on the test set, providing a decent start for predictive modeling efforts. To refine our model, I conducted a grid search specifically tuning the number of neighbors (k) for the KNeighborsClassifier, which slightly improved the performance, raising the accuracy to 0.79 on the test set.
# 
# Building on this initial model, I expanded my search to include Logistic Regression and RandomForest Classifier, exploring a range of hyperparameters for each. Through an extensive grid search encompassing different configurations, the Logistic Regression model with a regularization strength 
# C of 10 emerged as the most effective, achieving the highest cross-validation score of 0.81. This finding underscores the potential of Logistic Regression in handling binary classification problems like loan approval, where the balance between bias and variance is crucial for making accurate predictions.
# 
# Ultimately, when tested against the unseen data of the test set, the optimized Logistic Regression model demonstrated superior performance, recording an accuracy of 0.82. This result not only highlights the model's robustness but also its generalizability to new data, a critical aspect of any predictive model used in financial applications. The improvement in performance from the baseline KNN model to the tuned Logistic Regression model illustrates the importance of model selection and hyperparameter tuning in achieving higher prediction accuracy. These findings provide valuable insights for financial institutions looking to implement predictive analytics for loan approval decisions, suggesting that a well-tuned Logistic Regression model could significantly enhance decision-making processes.
# 
# 
# Conclusion:
# 
# Conclusions
# The analysis conducted on the loan approval dataset has successfully demonstrated the importance of careful model selection and hyperparameter tuning in the development of predictive models. Starting with a baseline KNeighborsClassifier model, which achieved an accuracy of 0.78 on the test set, I advanced my methodology through systematic hyperparameter tuning, which slightly enhanced the performance to an accuracy of 0.79. Further exploration using a broader set of models and more extensive parameter tuning led us to identify Logistic Regression with a regularization strength of 
# C = 10 as the optimal model, culminating in a commendable accuracy of 0.82 on the test set. This progression underscores the transformative impact of appropriate machine learning techniques in optimizing model performance, particularly in sectors such as finance where prediction accuracy is crucial.
# 
# Way Forward:
# 
# Moving forward, the deployment of the optimized Logistic Regression model into a real-world testing environment is crucial to validate its effectiveness in operational settings. Continuous improvement strategies should be implemented, including the monitoring and regular updating of the model with new data to ensure relevance and accuracy over time.
# 
# Ethical considerations and fairness in model application should also be prioritized to prevent bias, ensuring that decisions are equitable across all demographics. 
# 
# Finally, for the model to be truly effective in supporting decision-making, it should be integrated into existing loan processing systems with a user-friendly interface that provides interpretable and transparent results. This integration will help in building trust and enhancing the decision-making process, thereby increasing customer satisfaction and compliance with regulatory standards.
# 

# In[ ]:




