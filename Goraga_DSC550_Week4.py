#!/usr/bin/env python
# coding: utf-8

# Data Mining (DSC550-T301_2245_1)
# Assignement Week 4;
# 
# Author: Zemelak Goraga;
# 
# Date: 04/6/2024

# In[1]:


# 1. Import required library
import pandas as pd


# In[2]:


# 2. Load the dataset into a DataFrame
auto_mpg = pd.read_csv('auto-mpg.csv')


# In[3]:


# 2.1 Save the dataset as auto-mpg
#auto_mpg.to_csv('auto-mpg.csv', index=False)


# In[4]:


# 2.2 Display the first 5 rows of the dataset
print(auto_mpg.head())


# In[5]:


# 2.3 Data wrangling
# Remove the car name column
auto_mpg = auto_mpg.drop(columns=['car name'])


# In[6]:


# Check the data types
print(auto_mpg.dtypes)


# In[9]:


# Check for missing values
missing_values = auto_mpg.isnull().sum()
print("Missing Values:\n", missing_values)


# In[13]:


# 3. Prepare the data for modeling

# The 'horsepower' column has no missing values. 
# Convert horsepower column to numeric and replace non-numeric values with mean

auto_mpg['horsepower'] = pd.to_numeric(auto_mpg['horsepower'], errors='coerce')
auto_mpg['horsepower'].fillna(auto_mpg['horsepower'].mean(), inplace=True)


# In[32]:


# Convert horsepower column to numeric
# Check the data types
print(auto_mpg.dtypes)


# Answer for the Questions:
# 
# The horsepower column likely imported as a string due to data entry errors, missing values representation, export/import settings, inconsistent data sources, missing data conversion, or encoding issues. Careful inspection and data wrangling are necessary to ensure the correct data type for analysis.
# 

# In[ ]:





# In[14]:


# Now, let's handle categorical variables. dummy variables created for the 'origin' column.

# Check unique values in 'origin' column
print("\nUnique Values in 'origin' column:\n", auto_mpg['origin'].unique())


# In[15]:


# Create dummy variables for 'origin' column
auto_mpg = pd.get_dummies(auto_mpg, columns=['origin'], drop_first=True)


# In[16]:


# Confirm dummy variables creation
print("\nData After Creating Dummy Variables:\n", auto_mpg.head())


# In[18]:


# 4. Explore correlations
correlation_matrix = auto_mpg.corr()
print(correlation_matrix)


# Answer for the Question:
# 
# Features highly correlated with MPG (miles per gallon) likely include:
# Weight: Heavier vehicles tend to have lower fuel efficiency due to increased fuel consumption.
# Displacement: Larger engine displacements often result in lower MPG as they consume more fuel.
# Cylinders: More cylinders generally indicate higher power output, but they can also lead to decreased fuel efficiency.
# Horsepower: Higher horsepower engines typically consume more fuel, resulting in lower MPG.
# Acceleration: Faster acceleration may correlate with lower MPG due to increased fuel consumption.
# 
# These features directly affect the energy consumption of the vehicle and are likely to exhibit strong correlations with MPG.

# In[19]:


# 5. Visualize mpg versus weight
import matplotlib.pyplot as plt

plt.scatter(auto_mpg['weight'], auto_mpg['mpg'])
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('MPG vs Weight')
plt.show()


# In[20]:


# 6. Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X = auto_mpg.drop(columns=['mpg'])
y = auto_mpg['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


# 7. Train linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Instantiate and fit the model
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[26]:


X_train


# In[27]:


y_train


# In[22]:


# 8. Evaluate linear regression model
# Predict on training and test set
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)


# In[28]:


y_train_pred


# In[29]:


y_test_pred


# In[30]:


# Calculate R2 score, RMSE, and MAE
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)


# Answer for the Questions:
# 
# The R-squared (R2) value measures the proportion of the variance in the dependent variable (MPG) that is predictable from the independent variables (features).
# 
# A higher R2 value (close to 1) indicates that the model explains a larger proportion of the variance in the target variable, suggesting better predictive performance.
# Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) are both metrics used to evaluate the accuracy of the regression model's predictions.
# 
# RMSE represents the average difference between the actual and predicted values, with lower values indicating better performance.
# MAE represents the average absolute difference between the actual and predicted values, with lower values indicating better performance.
# Interpretation of results:
# 
# Training Set:
# 
# A high R2 value (close to 1) on the training set indicates that the model fits the data well and explains a large proportion of the variance in MPG.
# A low RMSE and MAE on the training set indicate that the model's predictions are close to the actual values with minimal error.
# Test Set:
# 
# A similar or slightly lower R2 value compared to the training set suggests that the model generalizes well to unseen data.
# A low RMSE and MAE on the test set indicate that the model's performance remains consistent when making predictions on new data.
# In summary, high R2 values, low RMSE, and low MAE on both training and test sets indicate that the regression model accurately predicts MPG and generalizes well to unseen data.

# In[24]:


# Print the results
print("Linear Regression Performance:")
print("Training R^2:", r2_train)
print("Testing R^2:", r2_test)
print("Training RMSE:", rmse_train)
print("Testing RMSE:", rmse_test)
print("Training MAE:", mae_train)
print("Testing MAE:", mae_test)


# In[25]:


# 9. Pick another regression model: I used Random Forest Regression model as an alternative to Linear Regression
from sklearn.ensemble import RandomForestRegressor

# Instantiate and fit the model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict on training and test set
y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

# Calculate R2 score, RMSE, and MAE
r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
rmse_train_rf = mean_squared_error(y_train, y_train_pred_rf, squared=False)
rmse_test_rf = mean_squared_error(y_test, y_test_pred_rf, squared=False)
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)

# Print the results
print("\nRandom Forest Regression Performance:")
print("Training R^2:", r2_train_rf)
print("Testing R^2:", r2_test_rf)
print("Training RMSE:", rmse_train_rf)
print("Testing RMSE:", rmse_test_rf)
print("Training MAE:", mae_train_rf)
print("Testing MAE:", mae_test_rf)


# In[ ]:





# Title: Predictive Modeling of Automobile Fuel Efficiency Using Machine Learning Algorithms
# 
# Summary:
# This report presents the development and evaluation of machine learning models to predict the fuel efficiency (miles per gallon) of automobiles based on various features. The dataset used for this study is the 'auto-mpg' dataset obtained from Kaggle, containing information such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin of the automobiles. Two regression models, namely Linear Regression and Random Forest Regression, were employed to predict fuel efficiency. The performance of these models was evaluated using metrics such as R-squared, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
# 
# 
# Introduction:
# Predicting fuel efficiency is crucial for automotive manufacturers and policymakers to develop more efficient vehicles and sustainable transportation systems. Machine learning techniques offer promising solutions to accurately predict fuel efficiency based on various vehicle characteristics. In this report, we explore the application of regression models to predict fuel efficiency using the auto-mpg dataset.
# 
# 
# Statement of the Problem:
# The primary objective of this study is to develop robust machine learning models capable of accurately predicting automobile fuel efficiency based on key features such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin. 
# 
# We aim to address the following research questions:
# 
# Can we accurately predict fuel efficiency using regression models?
# Which features are most influential in determining fuel efficiency?
# How do different regression algorithms compare in terms of predictive performance?
# 
# 
# Methodology:
# 
# Data Collection: The auto-mpg dataset was obtained from Kaggle, containing information about automobiles and their fuel efficiency.
# 
# Data Preprocessing:
# Data wrangling was performed to handle missing values and convert categorical variables into numerical format.
# Features such as car name were removed, and dummy variables were created for categorical features like origin.
# 
# Data Analysis (EDA):
# Correlation analysis was conducted to identify relationships between features and fuel efficiency.
# Visualization techniques were employed to gain insights into the dataset.
# 
# Model Development:
# Two regression models, Linear Regression and Random Forest Regression, were trained on the preprocessed data.
# The models were evaluated using R-squared, RMSE, and MAE metrics.
# Performance Evaluation:
# The performance of each model was assessed on both training and testing datasets to ensure generalization capability.
# 
# 
# Results:
# 
# The results obtained from the analysis provide valuable insights into the predictive modeling of automobile fuel efficiency using machine learning algorithms. Let's discuss and compare the findings from both the linear regression and random forest regression models:
# 
# 
# The correlation values between MPG and other features are as follows:
# 
# Cylinders: -0.775396
# Displacement: -0.804203
# Horsepower: -0.771437
# Weight: -0.831741
# Acceleration: 0.420289
# Model year: 0.579267
# Origin_2: 0.259022
# Origin_3: 0.442174
# 
# 
# Regression parameters:
# 
# 
# For linear regression:
# Training R^2: 0.8188288951042786
# Testing R^2: 0.8449006123776617
# Training RMSE: 3.370273563938906
# Testing RMSE: 2.8877573478836314
# Training MAE: 2.6054846937710368
# Testing MAE: 2.2875867704421067
# 
# 
# For random forest regression:
# Training R^2: 0.9810189898945959
# Testing R^2: 0.9105817015747857
# Training RMSE: 1.0908884599607205
# Testing RMSE: 2.1926476945692848
# Training MAE: 0.7477955974842765
# 
# 
# Linear Regression vs. Random Forest Regression:
# 
# Linear Regression:
# 
# Achieved a training R^2 of 0.8188 and testing R^2 of 0.8449.
# RMSE on the training set was 3.3703, while on the testing set, it was 2.8878.
# MAE on the training set was 2.6055, and on the testing set, it was 2.2876.
# Linear regression is a simple and interpretable model that assumes a linear relationship between the independent and dependent variables. It performs reasonably well, explaining around 82% of the variance in the training data and 84% in the testing data.
# 
# 
# Random Forest Regression:
# Achieved a training R^2 of 0.9810 and testing R^2 of 0.9106.
# RMSE on the training set was 1.0909, while on the testing set, it was 2.1926.
# MAE on the training set was 0.7478.
# Random forest regression is an ensemble learning method that builds multiple decision trees and averages their predictions. It provides higher accuracy and better generalization compared to linear regression, with an R^2 of approximately 91% on the testing data.
# Findings:
# 
# 
# Correlation Analysis:
# Features highly correlated with MPG include weight, displacement, cylinders, horsepower, and acceleration. These features directly affect fuel efficiency, with heavier vehicles, larger engine displacements, more cylinders, higher horsepower, and faster acceleration generally leading to lower MPG.
# 
# 
# Model Performance:
# Both models performed reasonably well, with the random forest regression model outperforming the linear regression model in terms of predictive accuracy.
# The linear regression model explained around 82% of the variance in the training data, while the random forest regression model explained approximately 98% of the variance.
# The random forest regression model demonstrated better generalization to unseen data, with an R^2 of approximately 91% on the testing data compared to 84% for linear regression.
# Comparison and Contrast:
# 
# 
# Complexity:
# Linear regression is a simple, interpretable model that assumes a linear relationship between variables. It may not capture complex nonlinear relationships in the data.
# Random forest regression is a more complex model that can capture nonlinear relationships and interactions between variables. It often provides higher predictive accuracy but may be less interpretable.
# 
# 
# Performance:
# Linear regression performs reasonably well but may underperform when the relationship between variables is nonlinear or when there are interactions between features.
# Random forest regression typically provides higher accuracy and better generalization to unseen data, making it more suitable for complex datasets with nonlinear relationships.
# 
# Interpretability:
# Linear regression provides straightforward interpretation of coefficients, making it easier to understand how each feature contributes to the prediction.
# Random forest regression, while more accurate, is less interpretable due to its ensemble nature and the complexity of multiple decision trees.
# 
# In summary, the findings indicate that while both linear regression and random forest regression models can be effective for predicting automobile fuel efficiency, the random forest regression model offers better predictive performance, particularly in capturing complex relationships in the data. However, the choice between the two models depends on factors such as the desired level of interpretability and the trade-off between accuracy and simplicity
# 
# 
# 
# Discussion:
# 
# The analysis of predictive modeling for automobile fuel efficiency using machine learning algorithms revealed valuable insights into the relationship between various vehicle features and miles per gallon (MPG). 
# 
# The correlation analysis of the dataset revealed significant relationships between MPG (miles per gallon) and various vehicle features. Notably, MPG exhibited strong negative correlations with weight (-0.83), displacement (-0.80), cylinders (-0.78), and horsepower (-0.77), indicating that heavier vehicles with larger engine displacements, more cylinders, and higher horsepower tend to have lower fuel efficiency. Additionally, acceleration showed a weaker positive correlation with MPG (0.42), suggesting that faster acceleration may lead to decreased fuel efficiency. Furthermore, model year exhibited a moderate positive correlation with MPG (0.58), indicating that newer vehicles tend to have better fuel efficiency. Finally, dummy variables representing the origin of the vehicle also displayed correlations with MPG, with origin_2 (0.26) and origin_3 (0.44) indicating positive relationships, although relatively weaker compared to other features. 
# 
# 
# The comparison between linear regression and random forest regression models highlighted distinct performance characteristics: while linear regression provided reasonable predictive accuracy, the random forest regression model outperformed it, achieving a higher R^2 value on both the training and testing datasets. Specifically, the random forest regression model demonstrated superior predictive accuracy, explaining approximately 91% of the variance in MPG on the testing data, compared to 84% for linear regression, showcasing its effectiveness in capturing complex relationships and enhancing predictive performance for automobile fuel efficiency modeling.
# 
# 
# Conclusion:
# 
# In conclusion, the analysis of the automobile fuel efficiency dataset using machine learning algorithms yielded valuable insights. The correlations highlighted the significant impact of various vehicle attributes on MPG. Notably, heavier vehicles with larger engine displacements, more cylinders, and higher horsepower exhibited lower fuel efficiency, while faster acceleration also correlated with decreased MPG. Conversely, newer vehicles tended to have better fuel efficiency, as indicated by the positive correlation with model year. Moreover, the origin of the vehicle, represented by dummy variables, showed modest correlations with MPG. The linear regression and random forest regression models provided effective means for predicting fuel efficiency, with both demonstrating strong performance metrics on both training and test datasets. Overall, these findings underscore the importance of considering vehicle weight, engine specifications, acceleration, model year, and origin when designing and evaluating strategies to improve automobile fuel efficiency.
# 
# 
# Way Forward:
# 
# Moving forward, further research could delve into exploring additional factors that influence automobile fuel efficiency, such as aerodynamics, tire characteristics, and driving conditions. Additionally, incorporating more advanced machine learning techniques like gradient boosting and neural networks could enhance predictive accuracy. Moreover, conducting a comparative analysis with alternative fuel vehicles or hybrid models could provide insights into the efficacy of different propulsion systems in achieving higher MPG ratings. Furthermore, ongoing advancements in automotive technology and data analytics offer opportunities to develop more sophisticated models for optimizing fuel efficiency and reducing environmental impact. Collaboration with automotive manufacturers and policymakers can facilitate the implementation of findings into real-world applications, fostering the development of more sustainable transportation solutions.

# In[ ]:




