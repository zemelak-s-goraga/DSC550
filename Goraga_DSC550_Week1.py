#!/usr/bin/env python
# coding: utf-8

# ## Data Mining (DSC550-T301_2245_1)
# 
# Assignement Week 1;
# Author: Zemelak Goraga;
# Date: 03/16/2024

# In[1]:


# Step 1: Import necessary libraries
import pandas as pd


# In[4]:


# Step 2: Load the dataset into a Pandas DataFrame and save it as df
df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")


# In[5]:


# Step 3: Display the first ten rows of the df dataset
print("First ten rows of the dataset:")
print(df.head(10))


# In[6]:


# Step 4: Display the columns of the df dataset
print("\nColumns of the dataset:")
print(df.columns)


# In[7]:


# Step 5: Find the dimensions (number of rows and columns) in the df data frame
num_rows, num_cols = df.shape
print(f"\nDimensions of the dataset: {num_rows} rows x {num_cols} columns")


# In[9]:


# Step 6: Find the top five games by critic score
top_games_by_critic_score = df.nlargest(5, 'Critic_Score')[['Name', 'Critic_Score']]
print("\nTop five games by critic score:")
print(top_games_by_critic_score)


# In[11]:


# Step 7: Find the number of video games in the df data frame in each genre
genre_counts = df['Genre'].value_counts()
print("\nNumber of video games in each genre:")
print(genre_counts)


# In[13]:


# Step 8: Find the first five games in the df data frame on the SNES platform
snes_games = df[df['Platform'] == 'SNES'].head(5)
print("\nFirst five games on the SNES platform:")
print(snes_games)


# In[15]:


# Step 9: Find the five publishers with the highest total global sales
publisher_sales = df.groupby('Publisher')['Global_Sales'].sum().nlargest(5)
print("\nFive publishers with the highest total global sales:")
print(publisher_sales)


# In[16]:


# Step 10: Create a new column for the percentage of global sales from North America
df['NA_Sales_Percentage'] = (df['NA_Sales'] / df['Global_Sales']) * 100


# In[17]:


# Step 11: Display the first five rows of the new DataFrame
print("\nFirst five rows with the new column:")
print(df.head())


# In[18]:


# Step 12: Find the number of NaN entries in each column
nan_counts = df.isna().sum()
print("\nNumber of NaN entries in each column:")
print(nan_counts)


# In[19]:


# Step 13: Replace non-numerical user score entries with NaN
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')


# In[21]:


# Step 14: Calculate the median user score
median_user_score = df['User_Score'].median()
median_user_score


# In[22]:


# Step 15: Replace NaN entries in the user score column with the median value
df['User_Score'].fillna(median_user_score, inplace=True)


# In[24]:


# Step 16: Display the updated DataFrame
print("\nUpdated DataFrame with NaN replaced:")
print(df)


# In[ ]:





# Title: Comprehensive Analysis of Video Game Sales with Ratings Dataset
# 
# Summary:
# This report provides a comprehensive analysis of the Video Game Sales with Ratings dataset, focusing on various aspects such as top games by critic score, genre distribution, publisher sales, and user score data wrangling. Through thorough examination and analysis, valuable insights into the video game industry are derived, aiding stakeholders in making informed decisions.
# 
# Introduction:
# The video game industry has witnessed exponential growth over the years, with the rise of various platforms and genres catering to diverse audiences. Understanding the dynamics of this industry is crucial for stakeholders, including developers, publishers, and investors. The Video Game Sales with Ratings dataset offers a wealth of information that can be leveraged to gain insights into consumer preferences, market trends, and more.
# 
# Statement of the Problem:
# The dataset presents several challenges and opportunities for analysis. Key issues include missing data entries, inconsistent formats, and the need to derive meaningful insights from the available information. The goal is to extract actionable insights that can inform business strategies and decision-making processes.
# 
# Methodology:
# 
# Data Acquisition: The dataset was obtained from Kaggle using the Kaggle API.
# Data Preprocessing: Data cleaning and wrangling techniques were applied to handle missing values, format inconsistencies, and prepare the data for analysis.
# Exploratory Data Analysis (EDA): Various statistical and visual methods were employed to explore the dataset and uncover patterns, trends, and relationships.
# Data Analysis: Quantitative analysis techniques were used to derive insights into key metrics such as top games, genre distribution, publisher sales, and user scores.
# 
# Dimensions of the dataset: 16719 rows x 16 columns, representing the number of observations (video games) and studied variables about the games, respectively. 
# 
# 
# Results:
# 
# Top five games by critic score:
# 
# Grand Theft Auto IV - 98.0;
# Grand Theft Auto IV - 98.0;
# Tony Hawk's Pro Skater 2 - 98.0;
# SoulCalibur - 98.0;
# Grand Theft Auto V - 97.0
# 
# 
# Genre distribution:
# 
# Action: 3370;
# Sports: 2348;
# Misc: 1750;
# Role-Playing: 1500;
# Shooter: 1323;
# Adventure: 1303;
# Racing: 1249;
# Platform: 888;
# Simulation: 874;
# Fighting: 849;
# Strategy: 683;
# Puzzle: 580;
# 
# Publisher sales:
# 
# Nintendo: 1788.81;
# Electronic Arts: 1116.96;
# Activision: 731.16;
# Sony Computer Entertainment: 606.48;
# Ubisoft: 471.61
# 
# 
# User score data wrangling:
# 
# Median user score: 7.5
# NaN entries replaced
# 
# 
# Discussion of Results:
# 
# Top Games: The analysis reveals that Grand Theft Auto IV, Tony Hawk's Pro Skater 2, and SoulCalibur are among the top-rated games by critics, suggesting a strong market demand for immersive gaming experiences. This indicates potential opportunities for game developers and publishers to focus on creating high-quality titles that resonate with players.
# 
# Genre Distribution: Action games dominate the market, followed by sports and miscellaneous genres. This highlights the diverse preferences of gamers and underscores the importance of catering to various interests to maximize market reach and revenue potential. Developers may benefit from targeting specific genres based on consumer demand and emerging trends.
# 
# Publisher Sales: Nintendo emerges as the top publisher with the highest total global sales, emphasizing the significance of brand reputation and quality content in driving sales. Other leading publishers such as Electronic Arts and Activision also play a significant role in shaping the gaming landscape, indicating fierce competition within the industry.
# 
# User Score Data: The median user score of 7.5 reflects a generally positive reception among gamers, although challenges such as missing data entries necessitate robust data cleaning and preprocessing techniques. Addressing these issues is essential to ensure the accuracy and reliability of user score data, enabling stakeholders to make informed decisions based on consumer feedback.
# 
# Conclusions:
# The analysis of the Video Game Sales with Ratings dataset provides valuable insights into the video game industry, offering stakeholders a deeper understanding of market dynamics and consumer behavior. By leveraging these insights, stakeholders can make informed decisions to enhance product development, marketing strategies, and overall business performance.
# 
# Recommendations:
# 
# Data Quality Assurance: Implement robust data cleaning and validation processes to ensure data accuracy and consistency.
# Market Segmentation: Utilize genre preferences and user demographics to tailor marketing campaigns and product offerings.
# Investment Strategies: Consider partnering with top publishers or investing in genres with high market demand and potential for growth.
# Way Forward:
# Further research could focus on longitudinal analysis to track trends over time, sentiment analysis of user reviews to gauge consumer sentiment, and predictive modeling to forecast future sales trends and game popularity.
# 

# In[ ]:




