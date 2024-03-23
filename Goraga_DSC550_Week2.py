#!/usr/bin/env python
# coding: utf-8

# ## Data Mining (DSC550-T301_2245_1)
# 
# Assignement Week 2;
# 
# Author: Zemelak Goraga;
# 
# Date: 03/22/2024

# In[25]:


import pandas as pd


# In[26]:


# Code to import 'animals.csv' dataset which contain export quantity (headsof live animals) and export value (US$) of cattle, sheep, pigs, and chikens
animals = pd.read_csv('animals.csv')


# In[27]:


# Code to display the first 5 rows of the dataset
print(animals.head())


# In[28]:


# Filter the dataset where item = pigs and save it as pigs.csv
pigs = animals[animals['item'] == 'Pigs']
pigs.to_csv('pigs.csv', index=False)


# Formulate 3 measurable questions:
# 
# 1. How has the export quantity of pigs changed over the years?
# 2. What are the top countries exporting pigs in terms of export value?
# 3. Is there a correlation between the export quantity and value of pigs?

# In[29]:


# Code to inspect the pigs dataset
print(pigs.head())


# In[30]:


# Code to inspect the pigs dataset
print(pigs.tail())


# In[31]:


# Code to rename 'area' as 'country' and 'item' as 'animal_category'
pigs_copy = pigs.copy()  # Create a copy of the DataFrame
pigs_copy.rename(columns={'area': 'country', 'item': 'animal_category'}, inplace=True)

pigs_copy


# In[32]:


# Assume the data is not clean and perform data wrangling like removing null values
pigs_copy.dropna(inplace=True)


# In[33]:


# Descriptive Statistics of 'Export Quantity' and 'Export Value' of live pigs using the whole dataset
# Perform summary statistics of the pigs dataset by considering 'value' as dependent variable and grouped by 'element'
summary_statistics_by_element = pigs_copy.groupby('element')['value'].describe()

# Print the summary statistics for 'Export Quantity' and 'Export Value'
print("Summary Statistics by Element:")
print(summary_statistics_by_element)


# In[ ]:





# In[34]:


# Descriptive Statistics of 'Export Quantity' and 'Export Value' of live pigs by country (top 10)
# Filter the DataFrame to include only rows where 'element' is 'Export Quantity' or 'Export Value'
export_data = pigs_copy[pigs_copy['element'].isin(['Export Quantity', 'Export Value'])]

# Group the filtered DataFrame by country and 'element' and sum the values
top_10_countries = export_data.groupby('country')['value'].sum().nlargest(10).index
export_data_top_10_countries = export_data[export_data['country'].isin(top_10_countries)]

# Perform summary statistics of the pigs dataset by considering 'value' as dependent variable and grouped by country and 'element'
summary_statistics_by_country_element = export_data_top_10_countries.groupby(['country', 'element'])['value'].describe()

# Print the summary statistics for 'Export Quantity' and 'Export Value' by country for the top 10 countries
print("Summary Statistics by Country and Element (Top 10 Countries):")
print(summary_statistics_by_country_element)


# Descriptive Statistics - Whole Dataset:
# 
# For the entire dataset, the summary statistics reveal the following insights:
# 
# Export Quantity:
# 
# The dataset consists of 1,271 observations.
# The mean export quantity is approximately 2.83 million heads of live pigs.
# The standard deviation is approximately 4.69 million, indicating considerable variability in export quantities.
# The minimum export quantity observed is 100,700 heads of live pigs, while the maximum is 38,577,345 heads.
# Export Value:
# 
# There are 658 observations for export values.
# The mean export value is approximately $534,737,000.
# The standard deviation is approximately $685,018, indicating a wide range of export values.
# The minimum export value observed is $100,697, and the maximum is $5,029,732.
# These statistics provide an overview of the distribution of export quantities and values for the entire dataset, indicating substantial variability in both.
# 
# Descriptive Statistics - By Country (Top 10 Countries):
# 
# For the top 10 countries, the summary statistics provide insights into the distribution of export quantities and values for each country individually:
# 
# For each country, the statistics are presented separately for 'Export Quantity' and 'Export Value'.
# 'Count' indicates the number of observations for each country and element.
# 'Mean' represents the average export quantity or value for each country.
# 'Std' is the standard deviation, indicating the spread of values around the mean.
# 'Min' and 'Max' represent the minimum and maximum values observed.
# '25%', '50%', and '75%' are the quartiles, indicating the values below which a certain percentage of observations fall.
# These statistics help understand the variability and distribution of export quantities and values across the top 10 countries, providing valuable insights for analysis and decision-making.
# 

# In[35]:


import matplotlib.pyplot as plt

# Filter the DataFrame to include only rows where 'element' is 'Export Quantity'
export_quantity_data = pigs_copy[pigs_copy['element'] == 'Export Quantity']

# Filter the export quantity data to include only rows within the years 1998 to 2013
export_quantity_data_15_years = export_quantity_data[(export_quantity_data['year'] >= 1998) & (export_quantity_data['year'] <= 2013)]

# Group the filtered DataFrame by country and sum the export quantities
top_countries_quantity = export_quantity_data_15_years.groupby('country')['value'].sum().nlargest(10)

# Filter the export quantity data to include only the top 10 countries
export_quantity_top_10 = export_quantity_data_15_years[export_quantity_data_15_years['country'].isin(top_countries_quantity.index)]

# Group the filtered data by country and year, and sum the export quantities
top_countries_quantity_by_year = export_quantity_top_10.groupby(['year', 'country'])['value'].sum().unstack('country')

# Plotting the trend of export quantity of live pigs for the top 10 countries from 1998 to 2013
plt.figure(figsize=(12, 8))
for country in top_countries_quantity_by_year.columns:
    plt.plot(top_countries_quantity_by_year.index, top_countries_quantity_by_year[country], marker='o', label=country)

plt.title('Export Quantity of Live Pigs for Top 10 Countries (1998 to 2013)')
plt.xlabel('Year')
plt.ylabel('Export Quantity (Head)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[36]:


from scipy.stats import linregress

# Filter the DataFrame to include only rows where 'element' is 'Export Quantity'
export_quantity_data = pigs_copy[pigs_copy['element'] == 'Export Quantity']

# Group the filtered DataFrame by country and sum the export quantities
top_countries_quantity = export_quantity_data.groupby('country')['value'].sum().sort_values(ascending=False).head(10)

# Plotting the top countries exporting pigs in terms of export quantity
plt.figure(figsize=(10, 6))
top_countries_quantity.plot(kind='bar')
plt.title('Top Countries Exporting Pigs in Terms of Export Quantity')
plt.xlabel('Country')
plt.ylabel('Export Quantity (heads of live pigs)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Filter the DataFrame to include only rows where 'element' is 'Export Quantity'
export_quantity_data = pigs_copy[pigs_copy['element'] == 'Export Quantity']

# Group the filtered DataFrame by country and sum the export quantities
top_countries_quantity = export_quantity_data.groupby('country')['value'].sum().sort_values(ascending=False).head(10)

# Filter the export quantity data to include only the top 10 countries
export_quantity_top_10 = export_quantity_data[export_quantity_data['country'].isin(top_countries_quantity.index)]

# Group the filtered data by country and calculate descriptive statistics for export quantities
descriptive_statistics_by_country = export_quantity_top_10.groupby('country')['value'].describe()

# Display the descriptive statistics for export quantities by country for the top 10 countries
print("Descriptive Statistics for Export Quantity by Country (Top 10 Countries):")
print(descriptive_statistics_by_country)



# In[ ]:





# In[37]:


import matplotlib.pyplot as plt

# Filter the DataFrame to include only rows where 'element' is 'Export Value'
export_value_data = pigs_copy[pigs_copy['element'] == 'Export Value']

# Group the filtered DataFrame by country and sum the export values
top_countries_value = export_value_data.groupby('country')['value'].sum().sort_values(ascending=False).head(10)

# Filter the export value data to include only the top 10 countries
export_value_top_10 = export_value_data[export_value_data['country'].isin(top_countries_value.index)]

# Group the filtered data by country and calculate the mean export value for each country
mean_export_value_by_country = export_value_top_10.groupby('country')['value'].mean()

# Plotting the mean export value for each country
plt.figure(figsize=(10, 6))
mean_export_value_by_country.plot(kind='bar')
plt.title('Mean Export Value by Country (Top 10 Countries)')
plt.xlabel('Country')
plt.ylabel('Mean Export Value (US$)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()



# Filter the DataFrame to include only rows where 'element' is 'Export Value'
export_value_data = pigs_copy[pigs_copy['element'] == 'Export Value']

# Group the filtered DataFrame by country and sum the export values
top_countries_value = export_value_data.groupby('country')['value'].sum().sort_values(ascending=False).head(10)

# Filter the export value data to include only the top 10 countries
export_value_top_10 = export_value_data[export_value_data['country'].isin(top_countries_value.index)]

# Group the filtered data by country and calculate descriptive statistics for export values
descriptive_statistics_by_country_value = export_value_top_10.groupby('country')['value'].describe()

# Display the descriptive statistics for export values by country for the top 10 countries
print("Descriptive Statistics for Export Value by Country (Top 10 Countries):")
print(descriptive_statistics_by_country_value)


# In[ ]:





# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Assuming 'element' column contains both 'Export Quantity' and 'Export Value'

# Filter the DataFrame to include only 'Export Quantity' and 'Export Value' rows
export_quantity_data = pigs_copy[pigs_copy['element'] == 'Export Quantity']
export_value_data = pigs_copy[pigs_copy['element'] == 'Export Value']

# Merge 'Export Quantity' and 'Export Value' data into a single DataFrame
merged_data = pd.merge(export_quantity_data, export_value_data, on=['country', 'year'], suffixes=('_quantity', '_value'))

# Calculate regression parameters
slope, intercept, r_value, p_value, std_err = linregress(merged_data['value_quantity'], merged_data['value_value'])

# Plot scatter plot with regression line fit for the whole dataset
sns.lmplot(x='value_quantity', y='value_value', data=merged_data, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.title('Correlation between Export Quantity and Export Value for the Whole Dataset')
plt.xlabel('Export Quantity (Heads of Live Pigs)')
plt.ylabel('Export Value (US$)')
plt.grid(True)

# Add the regression line to the plot
plt.plot(merged_data['value_quantity'], slope * merged_data['value_quantity'] + intercept, color='blue')

# Print correlation coefficient, p-value, and parameters for the line of fit
print("Overall Correlation Coefficient:", r_value)
print("P-value:", p_value)
print("Regression Parameters:")
print("Slope:", slope)
print("Intercept:", intercept)
print("Standard Error:", std_err)

plt.show()


# The results of the correlation analysis and regression provide valuable insights into the relationship between Export Quantity (Heads of Live Pigs) and Export Value (US$) for the dataset under consideration:
# 
# Overall Correlation Coefficient: The correlation coefficient indicates a strong positive correlation between Export Quantity and Export Value, with a value of approximately 0.97. This suggests that as the quantity of live pigs exported increases, the total export value in US dollars also tends to increase. The high correlation coefficient indicates a close linear relationship between the two variables.
# 
# P-value: The p-value associated with the correlation coefficient is extremely small (close to zero), indicating that the observed correlation is statistically significant. In statistical terms, this means that it is highly unlikely to observe such a strong correlation between Export Quantity and Export Value by random chance alone, providing evidence to support the validity of the observed relationship.
# 
# Regression Parameters:
# 
# Slope: The slope of the regression line is approximately 0.1165. This value represents the rate of change in Export Value (US$) for a one-unit increase in Export Quantity (Heads of Live Pigs). In this case, for each additional head of live pigs exported, the total export value increases by approximately $0.1165.
# Intercept: The intercept of the regression line is approximately -47863.38. This value indicates the estimated Export Value (US$) when the Export Quantity is zero. However, in practical terms, it may not have a meaningful interpretation since it falls outside the range of realistic quantities.
# Standard Error: The standard error is a measure of the variability of the observed data points around the regression line. A smaller standard error indicates a better fit of the regression line to the data. In this case, the standard error is very small (0.0011), suggesting that the regression line provides a good fit to the observed data points.
# Overall, the results indicate a strong positive linear relationship between Export Quantity and Export Value, with statistical significance. This information can be valuable for decision-making in the context of pig export markets, providing insights into the expected value of exports given a certain quantity of live pigs exported.

# In[ ]:





# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Assuming 'element' column contains both 'Export Quantity' and 'Export Value'

# Filter the DataFrame to include only 'Export Quantity' and 'Export Value' rows
export_quantity_data = pigs_copy[pigs_copy['element'] == 'Export Quantity']
export_value_data = pigs_copy[pigs_copy['element'] == 'Export Value']

# Get the top 5 countries by total export value
top_countries = export_value_data.groupby('country')['value'].sum().sort_values(ascending=False).head(5).index

# Create subplots for each country
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.flatten()

# Iterate through the top 5 countries
for i, country in enumerate(top_countries):
    # Filter data for the current country
    country_quantity_data = export_quantity_data[export_quantity_data['country'] == country]
    country_value_data = export_value_data[export_value_data['country'] == country]

    # Merge 'Export Quantity' and 'Export Value' data into a single DataFrame
    merged_data_country = pd.merge(country_quantity_data, country_value_data, on=['country', 'year'], suffixes=('_quantity', '_value'))

    # Calculate regression parameters
    slope, intercept, r_value, p_value, std_err = linregress(merged_data_country['value_quantity'], merged_data_country['value_value'])

    # Plot scatter plot with regression line for the current country
    sns.regplot(x='value_quantity', y='value_value', data=merged_data_country, ax=axs[i], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    axs[i].set_title(f'Correlation for {country}')
    axs[i].set_xlabel('Export Quantity (Heads of Live Pigs)')
    axs[i].set_ylabel('Export Value (US$)')
    axs[i].grid(True)

    # Add the regression line to the plot
    axs[i].plot(merged_data_country['value_quantity'], slope * merged_data_country['value_quantity'] + intercept, color='blue')

    # Print correlation coefficient, p-value, and parameters for the line of fit
    print(f"Correlation Coefficient for {country}: {r_value}")
    print(f"P-value for {country}: {p_value}")
    print(f"Regression Parameters for {country}:")
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"Standard Error: {std_err}")
    
plt.tight_layout()
plt.show()


# World:
# 
# The correlation coefficient for the entire dataset is approximately 0.98, with a very small p-value (close to zero), indicating a very strong positive correlation between Export Quantity and Export Value globally. This suggests that as the quantity of live pigs exported increases, the total export value in US dollars also tends to increase significantly.
# Europe:
# 
# Within Europe, the correlation coefficient is slightly higher than that of the world, approximately 0.99, with a very small p-value, indicating an even stronger positive correlation between Export Quantity and Export Value. This implies that the relationship between the two variables is exceptionally strong within the European region.
# European Union (EU):
# 
# The correlation coefficient for the European Union is very close to that of Europe, approximately 0.99, with a very small p-value. This indicates a strong positive correlation between Export Quantity and Export Value within the EU, similar to the broader European region.
# Western Europe:
# 
# In Western Europe, the correlation coefficient is slightly lower than that of Europe, approximately 0.98, with a very small p-value. This still signifies a strong positive correlation within this region, suggesting that increases in Export Quantity tend to be associated with higher Export Values.
# Netherlands:
# 
# For the Netherlands, the correlation coefficient is slightly lower compared to broader regions, approximately 0.97, with a very small p-value. Despite the slight decrease in correlation coefficient, the relationship remains strong and statistically significant, indicating that increases in Export Quantity within the Netherlands are strongly associated with higher Export Values.
# In summary, these results demonstrate a consistently strong positive correlation between Export Quantity and Export Value across different regions and countries, with statistically significant relationships observed at both regional and country levels. Such findings can provide valuable insights for stakeholders in the pig export industry, aiding in market analysis, strategic planning, and decision-making processes.

# In[ ]:





# Report
# 
# Summary:
# This report analyzes the export data of pigs from a dataset containing information on export quantity and value of various animals. It aims to explore trends, patterns, and relationships within the pig export data.
# 
# Introduction:
# The global trade of pigs plays a significant role in the agricultural sector. Understanding the dynamics of pig exports, including quantity and value, is crucial for stakeholders in the industry. This report analyzes a dataset containing export data of pigs to derive insights that can inform decision-making processes.
# 
# Statement of the Problem:
# The primary objective is to analyze the export data of pigs to uncover trends, patterns, and correlations. Key areas of investigation include changes in export quantity over time, identifying top exporting countries based on export value, and exploring the relationship between export quantity and value.
# 
# Methodology:
# The methodology involves importing the dataset, filtering it for pig-related data, performing data wrangling tasks such as renaming columns and handling missing values, and conducting exploratory data analysis. Graphical visualizations are employed to present the findings effectively.
# 
# 
# Result and Discussion:
# 
# Descriptive Statistics - Whole Dataset:
# 
# Export Quantity: The dataset consists of 1,271 observations with a mean export quantity of approximately 2.83 million heads of live pigs. The standard deviation is about 4.69 million, indicating considerable variability. The minimum export quantity observed is 100,700 heads, while the maximum is 38,577,345 heads.
# Export Value: There are 658 observations for export values, with a mean export value of around $534,737,000. The standard deviation is approximately $685,018, indicating a wide range of export values. The minimum export value observed is $100,697, and the maximum is $5,029,732.
# Descriptive Statistics - By Country (Top 10 Countries):
# 
# For each of the top 10 countries, statistics are presented separately for 'Export Quantity' and 'Export Value'. These include count, mean, standard deviation, minimum, maximum, and quartiles. This provides insights into the variability and distribution of export quantities and values across the top exporting countries.
# Correlation Analysis:
# 
# The correlation analysis between Export Quantity and Export Value for the whole dataset reveals a strong positive correlation with a correlation coefficient of approximately 0.97. The p-value is close to zero, indicating statistical significance.
# The regression analysis shows that for every unit increase in Export Quantity (heads of live pigs), the Export Value increases by approximately $0.1165. The intercept, although less interpretable in practical terms, provides an estimated Export Value when the Export Quantity is zero.
# Correlation Analysis by Country:
# 
# Further correlation analyses were conducted for the top 5 countries by total export value. All countries exhibited strong positive correlations between Export Quantity and Export Value, with correlation coefficients ranging from approximately 0.97 to 0.99. These correlations were statistically significant with very low p-values.
# Visualization:
# 
# Scatter plots with regression lines were generated to visually represent the correlation between Export Quantity and Export Value for the whole dataset and for each of the top 5 countries. These visualizations provide a clear understanding of the relationship between the variables.
# Overall, the analysis indicates a strong positive relationship between Export Quantity and Export Value for live pigs, both for the entire dataset and for individual countries. This information can be valuable for stakeholders in the pig export industry for decision-making, market analysis, and strategic planning.
# 
# 
# Conclusion:
# In conclusion, the analysis provides valuable insights into the export quantity and value of live pigs, both at the aggregate and country levels. These insights equip stakeholders with the necessary information to navigate the complexities of the global pig trade effectively. Continued monitoring and analysis of export data are essential for adapting to changing market conditions and maximizing opportunities in the agricultural sector.
# 
# Way Forward:
# Moving forward, it is recommended to further explore factors influencing pig export dynamics, such as market demand trends, trade agreements, and regulatory policies. Additionally, future research could delve into the impact of external factors, such as disease outbreaks or climate change, on pig exports and identify strategies to mitigate associated risks. Continual monitoring and analysis of export data will remain crucial for informed decision-making and sustainable growth in the agricultural industry.

# In[ ]:




