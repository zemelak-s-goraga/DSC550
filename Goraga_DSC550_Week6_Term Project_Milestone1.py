#!/usr/bin/env python
# coding: utf-8

# ## Data Mining (DSC550-T301_2245_1)
# 
# Assignement Week 6: Term Project Milestone 1;
# 
# Author: Zemelak Goraga;
# 
# Date: 04/20/2024

# Topic of Term Project: Trend analysis on Meat Animals Export Marketing of Different Countries.
# 
# Miestone 1: Data Exploration and Problem Statement
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

# # Data Wrangling, Descriptive Statistics, and Visualizations

# In[2]:


# Importing the dataset:

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
df = pd.read_csv('live_animals_export.csv') # this 'live_animals_export.csv' dataset is part of the huge 'FAOSTAT' dataset which I downloaded previously using Kaggle API commands in Google Colab environment


# In[3]:


# Inspecting the dataset:

print(df.head())


# In[4]:


# Inspecting the dataset:

print(df.tail())


# In[5]:


# Inspecting the dataset:

print(df.info())


# In[ ]:





# In[6]:


# Renaming columns:

df.rename(columns={'area': 'country', 'item': 'animal_type'}, inplace=True)
df.head()


# In[7]:


# Data wrangling:

# Handling missing values
df.dropna(inplace=True)


# In[8]:


# Data wrangling:

# Handling duplicate rows
df.drop_duplicates(inplace=True)


# In[9]:


# Data wrangling:

# Handling inconsistent values
df['animal_type'] = df['animal_type'].str.lower()


# In[10]:


# Descriptive statistics of export quntity - considering the whole dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_quantity_df = df[df['element'] == 'Export Quantity']

# Descriptive Statistics
print("Descriptive Statistics for Export Quantity:")
print(export_quantity_df['value'].describe())


# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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





# In[18]:


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


# In[19]:


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





# In[20]:


# Descriptive statistics of 'Export Value' expressed as number of heads of live animals exported - using the whole dataset

import seaborn as sns
import matplotlib.pyplot as plt

# Filter the DataFrame based on the criteria
export_value_df = df[df['element'] == 'Export Value']

# Descriptive Statistics
print("Descriptive Statistics for Export Value:")
print(export_value_df['value'].describe())


# In[ ]:





# In[21]:


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





# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


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


# In[27]:


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





# In[28]:


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


# In[29]:


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




