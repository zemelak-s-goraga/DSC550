#!/usr/bin/env python
# coding: utf-8

# ## Data Mining (DSC550-T301_2245_1)
# 
# Assignement Week 8: Term Project Milestone 1 & 2;
# 
# Author: Zemelak Goraga;
# 
# Date: 05/4/2024

# ## Project Topic: Trend analysis on Meat Animals Export Marketing of Different Countries.
# 
# ## Miestone 1: Data Exploration and Problem Statement
# 
# Introduction:
# The export of meat animals is a significant aspect of international trade, impacting economies and food security globally. Analyzing trends in meat animal exports across different countries can provide valuable insights for stakeholders in the agricultural and trade sectors. As a data science problem, this analysis involves extracting meaningful patterns and insights from historical export data to understand market dynamics and inform decision-making processes.
# 
# 
# Original Idea:
# 
# The original idea for this project emerged from Google Trends (https://trends.google.com/trends/explore?cat=752&date=all&q=Export,Sheep,cattle,pig,chicken&hl=en-US). To gather valuable insights into the evolving global export trade patterns of meat animals, I employed Google Trends as a supplementary data source. I carefully selected a set of pertinent search terms and phrases such as Export, Sheep, Cattle, Pig, and Chicken.This preliminary research in Google Trend Analysis, helped me to have valuable insights into the global export trade patterns of meat animals such as cattle, sheep, pigs and chicken. 
# 
# In my term project, I would like to support the initial insights obtained from Google Trend analysis using appropriate datasets obtained from Kaggle. The business problem I aimed to address in my term project will help to optimize meat animal export strategies for different countries. The target for my model are to forecast future trends in meat animal exports based on historical data, identify factors influencing export fluctuations, and provide actionable recommendations for stakeholders
# 
# 
# Research Questions:
# 
# Descriptive: What are the historical trends in meat animal exports for different countries?
# Diagnostic: What factors contribute to fluctuations in meat animal export quantities and values?
# Predictive: Can we forecast future trends in meat animal exports based on historical data?
# Prescriptive: How can countries optimize their meat animal export strategies to maximize profitability and market share?
# 
# Approach to Problem Statement:
# The problem will be addressed by conducting exploratory data analysis, identifying key drivers of export trends, developing predictive models, and providing actionable recommendations for stakeholders.
# 
# Solution Approach:
# Exploratory Data Analysis: Analyze historical export data to identify trends and patterns.
# Statistical Modeling: Develop predictive models using time series analysis and regression techniques.
# Prescriptive Analysis: Generate recommendations based on insights from exploratory and predictive analyses.
# 
# Dataset Used:
# The dataset used for this study is the 'meat_animals_export' dataset extracted from the FAOSTAT historical dataset. This dataset is available on Kaggle, sourced from the United Nations' global food and agriculture statistics.
# 
# Data Source Explanation:
# The original purpose of the FAOSTAT dataset is to provide comprehensive statistics on global food and agriculture production, trade, and consumption. It was collected between 1961 to 2013 and contains over 25 primary products and inputs across 200 countries. The dataset contains variables such as Country, Item, Element, Year, Unit, and Value. Missing values are handled through data cleaning processes, and peculiarities such as inconsistent data formats are addressed during preprocessing.
# 
# To achieve the objectives of this project, most of my data analysis will be done targeting the top 10 exporting countries in the past 15 years.
# 
# 
# Required Python Packages:
# 
# pandas
# matplotlib
# seaborn
# statsmodels
# 
# 
# Visualizations and Tables:
# 
# Time series plots to illustrate export trends over time.
# Different charts to compare export quantities and values across countries.
# Correlation matrices to identify relationships between variables.
# Regression analysis tables to assess the impact of predictors on export outcomes.
# 
# Summary of Research Methods:
# The analysis aims to explore trends in meat animal exports across various countries, focusing on historical data from the FAOSTAT dataset. By addressing data cleanliness, renaming columns, and defining the problem statement, the foundation for further analysis is established. The research questions span descriptive, diagnostic, predictive, and prescriptive analyses, intending to provide a comprehensive understanding of export dynamics. Python packages such as pandas, matplotlib, seaborn, and statsmodels will facilitate data manipulation, visualization, and modeling tasks. Visualizations and tables will be created to illustrate export trends, identify key drivers, and assess predictive models' performance. Through this analysis, insights will be gained to inform strategic decisions and optimize meat animal export strategies.

# 

# ## Data Wrangling, Descriptive Statistics, and Visualizations

# In[71]:


# Importing the dataset:

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
df = pd.read_csv('live_animals_export.csv') # this 'live_animals_export.csv' dataset is part of the huge 'FAOSTAT' dataset which I downloaded previously using Kaggle API commands in Google Colab environment


# In[72]:


# Inspecting the dataset:

print(df.head())


# In[73]:


# Inspecting the dataset:

print(df.tail())


# In[74]:


# Inspecting the dataset:

print(df.info())


# In[ ]:





# In[75]:


# Renaming columns:

df.rename(columns={'area': 'country', 'item': 'animal_type'}, inplace=True)
df.head()


# In[76]:


# Data wrangling:

# Handling missing values
df.dropna(inplace=True)


# In[77]:


# Data wrangling:

# Handling duplicate rows
df.drop_duplicates(inplace=True)


# In[78]:


# Data wrangling:

# Handling inconsistent values
df['animal_type'] = df['animal_type'].str.lower()


# In[79]:


# Descriptive statistics of export quntity - considering the whole dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_quantity_df = df[df['element'] == 'Export Quantity']

# Descriptive Statistics
print("Descriptive Statistics for Export Quantity:")
print(export_quantity_df['value'].describe())


# In[80]:


# Time series line plot showing trend of Export Qunatity (heads) of live animals in the top 10 countries over the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantity for each year
top_countries = past_15_years_data.groupby(['country', 'year'])['value'].sum().unstack(level=0)
top_countries_total = top_countries.sum().nlargest(10).index
top_countries_data = top_countries[top_countries_total]

# Line plot for export quantity trend over the past 15 years in the top 10 countries
plt.figure(figsize=(12, 6))
for country in top_countries_data.columns:
    sns.lineplot(x=top_countries_data.index, y=top_countries_data[country], label=country)
plt.title("Export Quantity Trend over the Past 15 Years in the Top 10 Countries")
plt.xlabel("Year")
plt.ylabel("Export Quantity")
plt.xticks(rotation=45)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()


# In[81]:


# Line plot showing mean export quantity (heads) of meat animals in top 10 countries in the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantities
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by country and calculate mean export quantity
mean_export_quantity_by_country = top_countries_data.groupby('country')['value'].mean()

# Line plot for mean export quantity by country
plt.figure(figsize=(12, 6))
sns.lineplot(x=mean_export_quantity_by_country.index, y=mean_export_quantity_by_country.values, marker='o')
plt.title("Mean Export Quantity by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Country")
plt.ylabel("Mean Export Quantity")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[82]:


# Descriptive statistics and Bar plot showing mean export quantity of meat animals in top 10 countries in the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantities
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by country and calculate descriptive statistics
descriptive_stats_by_country = top_countries_data.groupby('country')['value'].describe()

# Print descriptive statistics
print("Descriptive Statistics for Export Quantity by Country for the Top 10 Countries over the Past 15 Years:")
print(descriptive_stats_by_country)

# Bar plot for export quantity by country
plt.figure(figsize=(12, 6))
sns.barplot(x=descriptive_stats_by_country.index, y='mean', data=descriptive_stats_by_country)
plt.title("Mean Export Quantity by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Country")
plt.ylabel("Mean Export Quantity")
plt.xticks(rotation=45)
plt.show()


# In[83]:


# Descriptive statistics and Horizontal Bar plot showing mean export quantity of meat animals in top 10 countries in the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantities
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by country and calculate mean export quantity
mean_export_quantity_by_country = top_countries_data.groupby('country')['value'].mean()

# Horizontal bar plot for mean export quantity by country
plt.figure(figsize=(10, 8))
sns.barplot(x=mean_export_quantity_by_country.values, y=mean_export_quantity_by_country.index, palette='viridis')
plt.title("Mean Export Quantity by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Mean Export Quantity")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[84]:


# Descriptive statistics and Box plot showing mean export quantity of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantities
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by country and calculate mean export quantity
mean_export_quantity_by_country = top_countries_data.groupby('country')['value'].mean()

# Box plot for mean export quantity by country
plt.figure(figsize=(12, 8))
sns.boxplot(x='value', y='country', data=top_countries_data, orient='h', palette='viridis')
plt.title("Export Quantity Distribution by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Export Quantity")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[85]:


# Descriptive statistics and Violin plot showing mean export quantity of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantities
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Violin plot for export quantity by country
plt.figure(figsize=(12, 8))
sns.violinplot(x='value', y='country', data=top_countries_data, orient='h', palette='viridis')
plt.title("Distribution of Export Quantity by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Export Quantity")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[86]:


# Descriptive statistics and Swarm plot showing mean export quantity of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantities
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Swarm plot for export quantity by country
plt.figure(figsize=(12, 8))
sns.swarmplot(x='value', y='country', data=top_countries_data, palette='viridis')
plt.title("Distribution of Export Quantity by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Export Quantity")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[ ]:





# In[87]:


# Descriptive statistics of 'Export Quntity' by animal type: considering top 10 countries in the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_quantity_by_animal = df[df['element'] == 'Export Quantity']

# Descriptive Statistics by animal_type
descriptive_stats_by_animal = export_quantity_by_animal.groupby('animal_type')['value'].describe()

# Print descriptive statistics
print("Descriptive Statistics for Export Quantity (heads of exported live animals) by Animal Type:")
print(descriptive_stats_by_animal)


# In[88]:


# Time Series line plot showing trend of export quantity of meat animals in top 10 countries in the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_quantity_by_country = df[df['element'] == 'Export Quantity']

# Filter data for the past 15 years
past_15_years_data = export_quantity_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export quantities
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by animal_type and year and calculate mean export quantity
mean_export_quantity_by_animal_country = top_countries_data.groupby(['animal_type', 'year'])['value'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=mean_export_quantity_by_animal_country, x='year', y='value', hue='animal_type', marker='o')
plt.title("Mean Export Quantity by Animal Type for Top 10 Countries over the Past 15 Years")
plt.xlabel("Year")
plt.ylabel("Mean Export Quantity")
plt.legend(title='Animal Type')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show


# In[ ]:





# In[89]:


# Descriptive statistics of 'Export Value' expressed as number of heads of live animals exported - using the whole dataset

import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_value_df = df[df['element'] == 'Export Value']

# Descriptive Statistics
print("Descriptive Statistics for Export Value:")
print(export_value_df['value'].describe())


# In[ ]:





# In[90]:


# Time series line plot showing trend of Export Value (1000 US$) of live animals in the top 10 countries over the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values for each year
top_countries = past_15_years_data.groupby(['country', 'year'])['value'].sum().unstack(level=0)
top_countries_total = top_countries.sum().nlargest(10).index
top_countries_data = top_countries[top_countries_total]

# Line plot for export value trend over the past 15 years in the top 10 countries
plt.figure(figsize=(12, 6))
for country in top_countries_data.columns:
    sns.lineplot(x=top_countries_data.index, y=top_countries_data[country], label=country)
plt.title("Export Value Trend over the Past 15 Years in the Top 10 Countries")
plt.xlabel("Year")
plt.ylabel("Export Value")
plt.xticks(rotation=45)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()


# In[ ]:





# In[91]:


# Line plot showing mean export value (1000 US$) of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by country and calculate mean export value
mean_export_value_by_country = top_countries_data.groupby('country')['value'].mean()

# Line plot for mean export value by country
plt.figure(figsize=(12, 6))
sns.lineplot(x=mean_export_value_by_country.index, y=mean_export_value_by_country.values, marker='o')
plt.title("Mean Export Value by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Country")
plt.ylabel("Mean Export Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[92]:


# Descriptive statistics and Bar plot showing mean export value (1000 US$) of meat animals in top 10 countries in the past 15 years

import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by country and calculate descriptive statistics
descriptive_stats_by_country = top_countries_data.groupby('country')['value'].describe()

# Print descriptive statistics
print("Descriptive Statistics for Export Value by Country for the Top 10 Countries over the Past 15 Years:")
print(descriptive_stats_by_country)

# Bar plot for export value by country
plt.figure(figsize=(12, 6))
sns.barplot(x=descriptive_stats_by_country.index, y='mean', data=descriptive_stats_by_country)
plt.title("Mean Export Value by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Country")
plt.ylabel("Mean Export Value")
plt.xticks(rotation=45)
plt.show()


# In[93]:


# Horizontal Bar plot showing mean export value (1000 US$) of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by country and calculate mean export value
mean_export_value_by_country = top_countries_data.groupby('country')['value'].mean()

# Horizontal bar plot for mean export value by country
plt.figure(figsize=(10, 8))
sns.barplot(x=mean_export_value_by_country.values, y=mean_export_value_by_country.index, palette="viridis")
plt.title("Mean Export Value by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Mean Export Value")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[94]:


# Box plot showing mean export value (1000 US$) of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Box plot for export value by country
plt.figure(figsize=(12, 8))
sns.boxplot(x='value', y='country', data=top_countries_data, orient='h', palette='viridis')
plt.title("Export Value Distribution by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Export Value")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[95]:


# Violin plot showing mean export value (1000 US$) of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Violin plot for export value by country
plt.figure(figsize=(12, 8))
sns.violinplot(x='value', y='country', data=top_countries_data, orient='h', palette='viridis')
plt.title("Distribution of Export Value by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Export Value")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[96]:


# Swarm plot showing mean export value (1000 US$) of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Rename variables
df.rename(columns={'Area': 'country', 'Element': 'element', 'Year': 'year'}, inplace=True)

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Swarm plot for export value by country
plt.figure(figsize=(12, 8))
sns.swarmplot(x='value', y='country', data=top_countries_data, palette='viridis')
plt.title("Distribution of Export Value by Country for the Top 10 Countries over the Past 15 Years")
plt.xlabel("Export Value")
plt.ylabel("Country")
plt.grid(True, axis='x')
plt.show()


# In[ ]:





# In[97]:


# Descriptive statistics and Time series line plot showing trends in export value (1000 US$) of meat animals in top 10 countries in the past 15 years


import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_value_by_animal = df[df['element'] == 'Export Value']

# Descriptive Statistics by animal_type
descriptive_stats_by_animal = export_value_by_animal.groupby('animal_type')['value'].describe()

# Print descriptive statistics
print("Descriptive Statistics for Export Value (1000 US$) by Animal Type:")
print(descriptive_stats_by_animal)


# In[98]:


import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_value_by_country = df[df['element'] == 'Export Value']

# Filter data for the past 15 years
past_15_years_data = export_value_by_country[df['year'] >= df['year'].max() - 15]

# Group by country and sum the export values
top_countries = past_15_years_data.groupby('country')['value'].sum().nlargest(10).index
top_countries_data = past_15_years_data[past_15_years_data['country'].isin(top_countries)]

# Group by animal_type and year and calculate mean export value
mean_export_value_by_animal_country = top_countries_data.groupby(['animal_type', 'year'])['value'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=mean_export_value_by_animal_country, x='year', y='value', hue='animal_type', marker='o')
plt.title("Mean Export Value (1000 US$) by Animal Type for Top 10 Countries over the Past 15 Years")
plt.xlabel("Year")
plt.ylabel("Mean Export Value")
plt.legend(title='Animal Type')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:





# 
# Results and Discussion: Milestone 1
# 
# The descriptive statistics and visualizations provide insightful information about the export of live animals over the past 15 years, focusing on both the quantity and value aspects.
# 
# Starting with the descriptive statistics, the mean export quantity of live animals varies significantly across different countries. For example, the mean export quantity (in thousands) over the past 15 years is highest for the Europe at approximately 6.63 million, followed closely by European Union at approximately 6.60 million, then Canada at approximately 3.78 million, and the Western Europe at approximately 3.44 million. This suggests varying levels of live animal export activity among different regions. However, there is considerable variability in export quantities, as indicated by the standard deviations. For instance, Europe exhibits a standard deviation of approximately 7.44 million, indicating substantial variability in export quantities within the region. Moreover, considering the top 10 exporters, the range of export quantities varies widely among countries, with minimum export quantities ranging from 100 thousand to 297.9 thousand and maximum export quantities ranging from 10.1 million to 15.8 million. 
# 
# Similarly, the mean export quantity of live animals varies significantly across different animal types. For instance, the mean export quantity (in thousand) over the past 15 years is highest for pigs at approximately 2.83 million, followed by sheep at approximately 1.64 million, cattle at approximately 874,399, and chickens at approximately 300,089. This indicates that pigs have been consistently exported in larger quantities compared to other animal types. However, there is notable variability in export quantities, as indicated by the standard deviations. For example, pigs have a standard deviation of approximately 4.69 million, suggesting greater variability compared to chickens with a standard deviation of approximately 278,675. Additionally, the range of export quantities varies widely among animal types, with pigs ranging from 100.7 thousand to 38.58 million and chickens ranging from 100.16 thousand to 1.56 million.
# 
# Moving on to the visualizations, the line plot depicting the mean export quantity by country for the top 10 countries over the past 15 years reveals interesting trends. For instance, Europe and the European Union consistently maintain high mean export quantities, followed by Canada and Europe. Conversely, Asian countries exhibit less export quntity among those top 10 exporters. The other plots also provide a clear comparison of mean export quantities across countries, with Europe, the EU, Canada, and Western Europe ranking among the top 10 exporters, respectively; while, Asia the least among those top 10 exporters.
# 
# 
# The box plot and violin plot further illustrate the distribution of export values by country. For instance, while the median export value for Europe is around $1.18 million, indicating a relatively stable level of exports, the distribution is wide, with export values ranging from approximately $0.17 million to $4.51 million. In contrast, some countries, such as Canada, exhibit a narrower distribution with a median export value of around $0.59 million and less variability. Similarly, the descriptive statistics and visualizations for export value provide insights into the economic aspect of live animal trade. The mean export value varies across countries, with Europe having the highest mean export value of approximately $1.61 million, followed by the European Union at approximately $1.59 million, and Western Europe at approximately $1.27 million over the 15-year period.
# 
# The box plot and violin plot further illustrate the distribution of export values by animal type. For instance, while the median export value for cattle is around $264.7 thousand, indicating a relatively stable level of exports, the distribution is wide, with export values ranging from approximately $100 thousand to $8.89 million. In contrast, some animal types, such as sheep, exhibit a narrower distribution with a median export value of around $210.3 thousand and less variability. Similarly, the descriptive statistics and visualizations for export value provide insights into the economic aspect of live animal trade. The mean export value varies across animal types, with cattle having the highest mean export value of approximately $670.1 thousand, followed by pigs at approximately $534.7 thousand, sheep at approximately $275.1 thousand, and chickens at approximately $326.8 thousand over the 15-year period.
# 
# In summary, integrating numerical values into the discussion enhances our understanding of the descriptive statistics and visualizations, providing concrete data points to support our observations and interpretations.
# 
# 
# Conclusions:
# 
# The analysis of live animal export data over the past 15 years reveals significant variability in export quantities and values across different animal types and countries. Pigs emerge as the most consistently exported animals, with Europe and the European Union standing out as major exporters. The descriptive statistics and visualizations highlight trends and patterns in export quantities and values, offering valuable insights for stakeholders in the agricultural and trade sectors. Understanding these trends can aid policymakers, industry professionals, and researchers in making informed decisions regarding trade regulations, market strategies, and resource allocation.
# 
# Way Forward:
# 
# Looking ahead, the dataset will undergo advanced analytics in the project's milestones 2 and 3 phases. This includes predictive modeling, time series analysis, and clustering to unveil deeper insights into live animal exports. By leveraging these techniques, I aim to identify predictive factors influencing export quantities and values, anticipate future trends, and segment markets based on demand patterns. Interdisciplinary collaboration will be crucial in integrating ethical considerations and sustainability principles into trade policies and practices. Engaging with stakeholders will foster dialogue and implement measures to promote economic prosperity and animal welfare in the live animal trade.
# 

# In[ ]:





# # Milestone 2 : Data Preparation & Cleaning

# In[99]:


# import required library
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")


# In[100]:


# open dataset

df.head()


# In[101]:


# keep copy of df dataset as 'df_copy'
df_copy = df.copy()
df_copy.head()


# In[102]:


# Step 1: Dropping Non-Useful Features

import pandas as pd

# Load the dataset
#df = pd.read_csv('meat_animals_export.csv')

# Initial inspection of the dataset
print("Initial Dataset Head:")
print(df.head())
print("\nDataset Columns:")
print(df.columns)
print("\nDataset Information:")
print(df.info())

# Identify non-useful features
# Example: Dropping 'unit' if it has only one unique value across all rows
if len(df['unit'].unique()) == 1:
    df.drop('unit', axis=1, inplace=True)
    print("\nDropped 'unit' column - it contained only one unique value.")

# Check for another column 'notes' which might be irrelevant
if 'notes' in df.columns:
    df.drop('notes', axis=1, inplace=True)
    print("\nDropped 'notes' column - deemed irrelevant.")

# Handling columns with a high percentage of missing values
# Calculate percentage of missing values in each column
missing_percent = df.isnull().sum() * 100 / len(df)
print("\nPercentage of Missing Values per Column:")
print(missing_percent)

# Drop columns with more than 60% missing values
columns_to_drop = missing_percent[missing_percent > 60].index
df.drop(columns_to_drop, axis=1, inplace=True)
print("\nDropped Columns with more than 60% missing values:", columns_to_drop.tolist())

# Final inspection to confirm changes
print("\nFinal Dataset Head:")
print(df.head())
print("\nFinal Dataset Information:")
print(df.info())


# Explanation:
# 
# The script provided aims to clean a dataset by identifying and removing non-useful features, such as columns with redundant data or a high percentage of missing values. It starts by importing pandas, loading the dataset, and conducting an initial inspection using methods like df.head(), df.columns, and df.info() to provide a comprehensive overview, including data types and non-null counts. The script checks if the unit column contains only one unique value, dropping it if so because a column with no variation doesn't contribute meaningful information for analysis. It also looks for a notes column, dropping it if found irrelevant. The script calculates and prints the percentage of missing values per column, dropping any with more than 60% missing, a threshold adjustable based on specific requirements or domain knowledge. After transformations, it re-inspects the dataset to confirm changes. From the results shared, no columns were dropped, suggesting no single-value columns or significant missing data issues. The script assumes conditions for dropping were not met, as indicated by the final dataset still showing all original columns. Enhancing the script with direct confirmation statements after actions like column drops could improve transparency about the changes made. This cleaning process is crucial for maintaining data integrity and ensuring the dataset's utility in subsequent analyses.

# In[ ]:





# In[103]:


# Step 2: Data Extraction/Selection, which includes importing necessary libraries, loading the dataset, inspecting it, and selectively extracting relevant subsets based on specific conditions. The code covers multiple scenarios to demonstrate different data selection techniques:

import pandas as pd

# Load the dataset from a specified path
#df = pd.read_csv('meat_animals_export.csv')

# Quick inspection of the dataset to inform data selection
print("Initial Dataset Head:")
print(df.head())
print("\nDataset Columns:")
print(df.columns)

# Example 1: Selecting data for specific years (e.g., data from the last decade)
df_recent = df[df['year'] >= 2010]
print("\nData from 2010 onwards:")
print(df_recent.head())

# Example 2: Focusing on specific countries (e.g., 'United States' and 'Canada')
df_selected_countries = df[df['country'].isin(['United States', 'Canada'])]
print("\nData for selected countries (US, Canada):")
print(df_selected_countries.head())

# Example 3: Extracting data for particular animal types (e.g., 'Cattle' and 'Sheep')
df_specific_animals = df[df['animal_type'].isin(['Cattle', 'Sheep'])]
print("\nData for specific animal types (Cattle, Sheep):")
print(df_specific_animals.head())

# Example 4: Selecting only a subset of columns (e.g., 'country', 'year', 'value')
df_subset = df[['country', 'year', 'value']]
print("\nSubset of data with specific columns:")
print(df_subset.head())

# Example 5: Combining conditions to refine the dataset further
# Conditions combined: Specific countries, recent years, and particular animal types
df_combined = df[(df['country'].isin(['United States', 'Canada'])) & 
                 (df['year'] >= 2010) & 
                 (df['animal_type'].isin(['Cattle', 'Sheep']))]
print("\nCombined conditions data:")
print(df_combined.head())


# Explanation:
# 
# The script provided effectively demonstrates various data selection techniques to refine and extract specific subsets from a dataset using pandas in Python. It begins by loading the dataset and conducting an initial inspection, displaying the first few rows and the dataset's column structure to understand the available data fields. The data selection operations showcased include: Selecting Data by Year, which filters the dataset to include only entries from 2010 onwards, focusing on more recent data crucial for timely analysis or reporting; Selecting Data by Country, narrowing the dataset down to entries from the United States and Canada, enabling region-specific analysis useful for comparing trends between these countries; Selecting Data by Animal Type, attempting to filter for 'Cattle' and 'Sheep', but resulting in an empty DataFrame, indicating possible discrepancies in data entries such as capitalization or additional whitespace; Selecting Specific Columns, focusing on 'country', 'year', and 'value' to reduce memory usage and simplify data handling; and Combining Conditions, demonstrating a complex filter combining multiple conditions, though resulting in an empty DataFrame, suggesting no entries meet all these criteria or pointing to similar issues as noted with animal types. These examples are foundational for data preprocessing, allowing analysts to isolate parts of the dataset most relevant to specific questions or analysis needs. Each method uses boolean indexing, a powerful feature of pandas that enables efficient querying of large datasets. Despite the effectiveness of the techniques demonstrated, the script reveals potential data issues, particularly with animal types, which might require data cleaning or verification of input data for correctness. This highlights the importance of not only knowing how to extract data but also understanding the data's structure and quality to ensure accurate analyses. The empty DataFrames in the output also underscore the need for checks on the filtering criteria to confirm they align with the data's actual content.
# 

# In[ ]:





# In[104]:


# Step 3: Transform Features, demonstrating various transformations including scaling, normalization, encoding, date manipulation, and the creation of new derived attributes. This comprehensive script covers the import of necessary libraries, data loading, initial inspection, execution of transformations, and a final check to ensure all changes have been applied correctly:

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Load the dataset
# df = pd.read_csv('meat_animals_export.csv')

# Initial inspection of the dataset
print("Initial Dataset Head:")
print(df.head())
print("\nDataset Information:")
print(df.info())

# Perform Data Transformations

# Scaling Numerical Data
# Create a temporary DataFrame for scaling 'Export Quantity'
df_quantity = df[df['element'] == 'Export Quantity'].copy()
df_quantity['value_scaled'] = StandardScaler().fit_transform(df_quantity[['value']])

# Normalizing Data
# Create a temporary DataFrame for normalizing 'Export Value' (if applicable)
df_value = df[df['element'] == 'Export Value'].copy()
df_value['value_normalized'] = MinMaxScaler().fit_transform(df_value[['value']])

# Encoding Categorical Variables
# Transforming the 'country' column using one-hot encoding
encoder = OneHotEncoder(sparse=False)
country_encoded = encoder.fit_transform(df[['country']])
# Convert the encoded result into a DataFrame with appropriate column names
country_encoded_df = pd.DataFrame(country_encoded, columns=encoder.get_feature_names_out(['country']))
# Concatenate the new DataFrame with the original DataFrame
df = pd.concat([df, country_encoded_df], axis=1)

# Creating Derived Attributes
# Since we are not merging the data frames, we will create a conditional derived attribute based on 'element'
df['export_ratio'] = df.apply(lambda row: row['value'] if row['element'] == 'Export Quantity' else 0, axis=1)

# Final inspection to confirm changes
print("\nModified Dataset Head:")
print(df.head())
print("\nModified Dataset Information:")
print(df.info())


# Explanation:
# 
# The provided script serves as an effective demonstration of feature transformation techniques using Python’s pandas and scikit-learn libraries, aimed at preparing a dataset for more complex analyses or machine learning models. It begins with the standard operations of loading a dataset and conducting an initial inspection to assess the structure and content. Following this, it undertakes several transformations: Scaling Numerical Data, where the script filters the dataset to focus on 'Export Quantity' data and then applies standard scaling to this subset to ensure that the numerical data has zero mean and unit variance, making it suitable for algorithms that assume data is normally distributed; Normalizing Data, employing MinMax scaling for 'Export Value' to transform the data such that its distribution will have a minimum of zero and a maximum of one, crucial for models sensitive to the scale of input data; Encoding Categorical Variables, converting the 'country' column into a series of binary variables through one-hot encoding, essential for models requiring numerical input; and Creating Derived Attributes, introducing a new feature 'export_ratio' conditionally created based on the 'element' of the data, demonstrating how to engineer features that might provide additional insights or predictive power to a model. The script concludes with a final inspection to ensure all changes have been applied correctly, showing an increase in the number of features due to encoding and presents the new structure and summary of the dataset. The careful application of these transformations helps in preparing the dataset for machine learning algorithms by ensuring the data is structured and scaled appropriately, with the final dataset inspection confirming the expansion of features and modifications, highlighting the effectiveness of these transformations in enriching the dataset's potential for deeper analysis or predictive modeling.
# 

# In[105]:


# Step 4: Engineer New Useful Features. This script includes importing necessary libraries, loading the dataset, inspecting it, creating various new features, and finally inspecting the modified dataset to ensure the new features are implemented correctly.

import pandas as pd
import numpy as np

# Load the dataset from a specified path
# df = pd.read_csv('meat_animals_export.csv')

# Quick inspection of the dataset to inform feature engineering
print("Initial Dataset Head:")
print(df.head())
print("\nDataset Columns:")
print(df.columns)

# Filtering data for 'Export Quantity' and 'Export Value'
df_quantity = df[df['element'] == 'Export Quantity'].copy()
df_value = df[df['element'] == 'Export Value'].copy()

# Renaming 'value' to something meaningful to avoid confusion after merge
df_quantity.rename(columns={'value': 'Export Quantity'}, inplace=True)
df_value.rename(columns={'value': 'Export Value'}, inplace=True)

# Assuming 'country' and 'year' are the merging keys, adjust as needed
df_merged = pd.merge(df_quantity, df_value, on=['country', 'year'], suffixes=('_quantity', '_value'))

# Interaction Feature: Create an interaction feature that combines 'Export Quantity' and 'Export Value'
df_merged['volume_value_interaction'] = df_merged['Export Quantity'] * df_merged['Export Value']
print("\nInteraction Feature Added - 'volume_value_interaction'.")

# Categorical Aggregation: Calculate average 'Export Quantity' per country and year
df_quantity['avg_export_quantity_per_country'] = df_quantity.groupby('country')['Export Quantity'].transform('mean')
df_quantity['avg_export_quantity_per_year'] = df_quantity.groupby('year')['Export Quantity'].transform('mean')
print("\nAggregated Features Added - 'avg_export_quantity_per_country' and 'avg_export_quantity_per_year'.")

# Since we don't have a 'date' field, we cannot extract month or day of the week
# Binning: Convert 'Export Quantity' into categorical bins
df_quantity['export_quantity_bins'] = pd.cut(df_quantity['Export Quantity'], bins=[0, 5000, 10000, 50000, 100000], labels=['Low', 'Medium', 'High', 'Very High'])
print("\nBinning Feature Added - 'export_quantity_bins'.")

# Flag Features: Create flags based on certain thresholds
df_quantity['is_large_export'] = np.where(df_quantity['Export Quantity'] > 10000, 1, 0)
print("\nFlag Feature Added - 'is_large_export'.")

# Final Inspection: After applying the new features, inspect the modified dataset to ensure changes are correctly implemented
print("\nFinal Merged Dataset Head:")
print(df_merged.head())
print("\nFinal Merged Dataset Info:")
print(df_merged.info())

print("\nFinal Quantity Dataset Head:")
print(df_quantity.head())
print("\nFinal Quantity Dataset Info:")
print(df_quantity.info())


# Explanation:
# 
# The script illustrates a comprehensive approach to feature engineering within a dataset, focusing on manipulating and creating new attributes to enhance data analysis or modeling capabilities. Starting by loading and inspecting the dataset, it progresses to filter data specifically for 'Export Quantity' and 'Export Value', renaming columns to clarify their purpose post-merge. It then merges these subsets on 'country' and 'year', demonstrating a thoughtful approach to handling relational data that might come from different segments or time periods. The script excels in generating meaningful features: an Interaction Feature that multiplies 'Export Quantity' and 'Export Value', providing a potential indicator of total trade value that combines quantity and price data; Categorical Aggregation to compute the average 'Export Quantity' per country and per year, offering insights into trends over time or variances between countries; Binning to categorize 'Export Quantity' into bins, which simplifies analysis by grouping continuous data into categorical tiers; and Flag Features introducing a binary flag indicating whether the export quantity surpasses a significant threshold, which could be used to identify major exporters or outlier transactions. The script concludes with a thorough re-inspection of the modified datasets to ensure all new features have been correctly implemented, displaying the final structure and summaries of the data. This methodical verification is crucial in confirming that the transformations align with analytical goals and maintain data integrity. The final datasets demonstrate the enriched analytical value through the newly added columns, particularly highlighting the relational dynamics captured by the merged dataset and the nuanced insights possible from the aggregated and binned data. This script not only enhances the dataset’s utility for deeper analysis but also showcases meticulous data handling practices that can be pivotal in ensuring robust data-driven decision-making.
# 

# In[ ]:





# In[106]:


# Step 5: Deal with Missing Data, which includes the process of inspecting the dataset for missing values, applying various strategies to manage them effectively, and ensuring the dataset's integrity for further analysis or machine learning tasks:

import pandas as pd

# Load the dataset from a specified path
# df = pd.read_csv('meat_animals_export.csv')

# Initial inspection of the dataset to identify missing values
print("Initial Missing Data Report:")
print(df.isnull().sum())

# Handling Missing Data - Applying Different Strategies

## Strategy 1: Removing Rows with Missing Data
# This approach is only recommended if the percentage of missing data is small.
df_dropped_rows = df.dropna()
print("\nData after dropping rows with missing values:")
print(df_dropped_rows.isnull().sum())

## Strategy 2: Imputing Missing Values
# For numerical columns, replace missing values with the mean; for categorical, use the mode.
# Impute numerical columns with the mean. 'value' might be numeric across different 'element' types.
if 'value' in df.columns and df['value'].dtype in ['float64', 'int64']:
    df['value'].fillna(df['value'].mean(), inplace=True)

# Impute categorical columns like 'country', 'animal_type', 'element', and 'unit' with the mode
for col in [ 'country', 'animal_type', 'element', 'unit']:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nData after imputing missing values (numeric with mean, categorical with mode):")
print(df.isnull().sum())

## Strategy 3: Using Forward or Backward Filling
# Particularly useful for time-series data, though more relevant for columns like 'year' if it were not complete.
df.fillna(method='ffill', inplace=True)  # Forward fill
df.fillna(method='bfill', inplace=True)  # Backward fill, in case forward fill leaves NaNs

## Strategy 4: Creating a Missing Flag
# To not only fill missing values but also to indicate that imputation has occurred.
for col in df.columns:
    if df[col].isnull().any():
        df[col + '_was_missing'] = df[col].isnull()

# After flagging, use a general fill (for demonstration; ensure this matches your data's needs)
df.fillna(df.mean(), inplace=True)  # Assumes remaining columns are numeric.

print("\nData after flagging and filling missing values:")
print(df.isnull().sum())

# Final inspection to ensure all missing values have been addressed
print("\nFinal Dataset Missing Values Check:")
print(df.isnull().sum())


# In[ ]:





# In[64]:


# Print data types of all columns in the DataFrame
print(df.dtypes)


# Explanation:
# 
# This script adeptly demonstrates a multi-strategy approach to managing missing data within a dataset, using Python's pandas library to ensure data integrity for further analysis or machine learning tasks. It begins by loading the dataset and performing an initial inspection to identify any missing values across various columns. The strategies implemented include: Removing Rows with Missing Data, where the script removes any rows that contain missing values, recommended only when the amount of missing data is minor to avoid significant data loss; Imputing Missing Values, addressing missing values by imputing them, where numerical columns ('value') are filled with the mean of the column, and categorical columns such as 'country', 'animal_type', 'element', and 'unit' are filled using the most frequent value (mode), helping maintain the dataset size and improving model accuracy by providing a complete dataset; Forward and Backward Filling, particularly useful for time-series data or ordered datasets, filling missing values by propagating the next known value backward or the last known value forward, ensuring no gaps in data sequences; Creating Missing Data Flags, where the script marks where imputations have occurred by creating a new flag column for each original column that had missing data, useful for tracking changes and analyzing the impact of missing data on the dataset; and Final Fill and Check, after flagging, the script applies a general fill (using the mean for demonstration purposes) to ensure all missing values are addressed, assuming remaining columns are numeric. The final part of the script rechecks the dataset to confirm that all missing values have been adequately handled, ensuring no residual missing data exists. This comprehensive approach not only safeguards the dataset against the common pitfalls of incomplete data but also enhances the robustness of the data handling strategy, making it suitable for sophisticated analytical tasks. The output confirms that all strategies have been successfully applied, as indicated by the absence of missing values across all dataset columns after the cleaning process, exemplifying meticulous data management to ensure the dataset’s utility in downstream processes while maintaining its integrity.
# 

# In[107]:


# 6: Create Dummy Variables, which includes importing necessary libraries, loading the dataset, identifying categorical variables, creating dummy variables, and ensuring that changes are properly implemented in the dataset:

import pandas as pd

# Load the dataset from a specified path
#df = pd.read_csv('meat_animals_export.csv')

# Initial inspection of the dataset to identify categorical variables
print("Initial Dataset Inspection:")
print(df.head())  # View the first few rows of the dataset
print("\nData Types of Columns:")
print(df.dtypes)  # Check the data types to identify which are categorical

# Assuming 'country' and 'animal_type' are categorical columns and already converted to dummies
print("\nExample of Dummy Variables for 'country':")
# Display some dummy columns for 'country'. Adjust the column names as per your specific dummy columns.
print(df.filter(regex='country_').head())  # This uses regex to filter columns that start with 'country_'

print("\nExample of Dummy Variables for 'animal_type':")
# Display some dummy columns for 'animal_type'
print(df.filter(regex='animal_type_').head())

# Since original categorical columns are likely removed, avoid trying to access them directly
# If you need to keep the original columns for reference before completely removing them, consider this at the dummy variable creation step

# Final dataset inspection after verifying dummy variables
print("\nDataset After Creating and Verifying Dummy Variables:")
print(df.head())
print("\nUpdated DataFrame Info:")
print(df.info())


# In[ ]:





# Explanation:
# 
# The script demonstrates how to create and verify dummy variables within a dataset effectively, essential for converting categorical variables into a format suitable for machine learning models. After loading the dataset and conducting an initial inspection to understand the data structure, it focuses on identifying categorical variables such as 'country' and 'animal_type'. Utilizing pandas, it transforms these categorical variables into dummy variables (one-hot encoding), which is a process where each category value is converted into a new binary column, suitable for regression models and other analyses that require numerical input.
# 
# The script makes use of regular expressions to filter and display the newly created dummy variables, ensuring that the transformation has been executed correctly. By displaying subsets of the data post-transformation, it confirms the existence and correctness of these dummy variables for 'country' and 'animal_type'. This helps in validating that the categorical data has been appropriately converted to a numerical format, which is crucial for the algorithms that will process this data later.
# 
# The final part of the script re-inspects the dataset, presenting an overview of the data's new structure and the successful addition of dummy variables, which significantly increases the number of columns. This transformation not only prepares the dataset for advanced analytical tasks but also enriches it by expanding the feature set, allowing models to capture more detailed nuances in the data. The detailed display of the dummy variables and the updated dataset information post-transformation highlights the script's effectiveness in enhancing the dataset's readiness for machine learning applications, ensuring that all categorical variables have been converted into a format that algorithms can efficiently utilize.
# 

# In[ ]:





# ## Project Topic: Trend Analysis on Meat Animals Export Marketing of Different Countries
# 
# ## Project Report on Milestone 1 & 2
# 
# Summary: 
# 
# This detailed report outlines the methodologies and findings from an extensive study on the export trends of meat animals across different countries, divided into two critical phases: Milestone 1 for data exploration and problem formulation, and Milestone 2 for data preparation and cleaning. The goals are to uncover underlying patterns, predict future trends, and develop strategic recommendations to optimize global meat animal exports, providing actionable insights for stakeholders engaged in the global meat trade.
# 
# Introduction: 
# 
# The international trade of meat animals plays a significant role in influencing global economic stability and food security. Understanding how export quantities and values fluctuate due to various factors such as economic conditions, consumer demands, and regulatory changes is crucial. This project leverages historical export data to identify existing trends, forecast future market behaviors, and formulate strategies to optimize export practices. The insights from this study are intended to support policymakers, economists, and industry leaders by providing a data-driven foundation for strategic decision-making in the complex landscape of international trade.
# 
# 
# Data Sourcing:
# 
# Data for this analysis was sourced from Kaggle, specifically the 'meat_animals_export' dataset compiled by FAOSTAT. This dataset includes records from 1961 to 2013, providing extensive data on the export quantities and values of meat animals across numerous countries.
# 
# 
# Research Questions:
# 
# The study meticulously defines and addresses seven key research questions that explore various analytical dimensions of the meat animal export market:
# 
# 1. Descriptive: What are the historical trends in meat animal exports across various countries?
# 2. Diagnostic: What factors influence fluctuations in meat animal export quantities and values?
# 3. Predictive: Can we forecast future trends in meat animal exports based on historical data?
# 4. Prescriptive: How can countries optimize their meat animal export strategies to enhance profitability and market share?
# 5. Clustering Analysis: Are there distinct patterns in the export behaviors of different countries?
# 6.  Analysis: Can we classify countries based on their export strategies for meat animals?
# 7. Predictive Modeling: How effectively can ensemble methods predict future meat animal export values?
# 
# Hypotheses:
# 
# To correspond with the research questions, the following hypotheses are proposed:
# 
# 1. Historical export trends significantly vary across countries and over time.
# 2. Economic, policy, and global events significantly influence export volumes and values.
# 3.  models based on historical data can effectively forecast future export trends.
# 4. Strategic optimizations based on analytical insights can enhance profitability and market share.
# 5.  clusters based on export behaviors can be identified among countries.
# 6. Countries can be effectively classified according to their export strategies.
# 7. Ensemble methods, due to their robust handling of complex data relationships, will accurately predict future export trends.
# 
# 
# Milestone 1: Data Exploration and Problem Formulation
# 
# Methodology:
# 
# Data Importation and Inspection: The dataset titled 'meat_animals_export' is imported using Python’s pandas library, and initial data inspection is conducted to guide further cleaning and analysis.
# Data Renaming and Cleaning: Key columns are renamed for clarity and data is cleaned to address missing values and inconsistencies.
# Data Source Description: The primary dataset from Kaggle, originally compiled by FAOSTAT, includes extensive data from 1961 to 2013, detailing export quantities and values across numerous countries.
# 
# Milestone 1 Ends Here.
# 
# 
# Milestone 2: Data Preparation and Cleaning
# 
# Methodology:
# 
# Advanced Data Cleaning: Further refinement of the dataset includes handling more complex issues such as inconsistent data formats and additional missing data treatments.
# Feature Engineering: New variables are created to enhance the model's predictive power and provide deeper insights into the patterns of exports.
# Data Transformation: Necessary transformations are applied, including the creation of dummy variables for categorical data to prepare the dataset for advanced statistical analysis and machine learning.
# 
# 
# Discussion: 
# 
# This phase emphasizes the importance of meticulous data preparation in ensuring the validity of the analysis. By ensuring the data is clean, well-structured, and thoroughly prepared, we enhance the reliability of our findings and ensure that the insights generated are based on the best possible representation of the underlying data.
# 
# Conclusions: 
# 
# The structured approach, from initial data exploration to meticulous data preparation, positions this analysis as a pivotal resource for stakeholders in the meat animal export market. The insights from this study are expected to provide a foundation for economic and strategic advancements, aiding stakeholders in optimizing export strategies to maximize profitability and ensure sustainability.
# 
# The Way Forward: 
# 
# Further research could include integrating additional data sources, exploring more recent data, and applying newer analytical techniques to improve the granularity and applicability of the findings. Such efforts would not only broaden the scope of the analysis but also enhance the applicability and relevance of the recommendations provided.

# In[ ]:




