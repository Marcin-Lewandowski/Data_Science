import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


url ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'

df = pd.read_csv(url)

# Analyzing Individual Feature Patterns Using Visualization

# list the data types for each column
#print(df.dtypes)


print(df['peak-rpm'].dtypes)

# we can calculate the correlation between variables of type "int64" or "float64" using the method "corr":
# Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.

df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


# Continuous Numerical Variables - Positive Linear Relationship. Let's find the scatterplot of "engine-size" and "price".

# Engine size as potential predictor variable of price


'''
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)



# Display 
plt.show()
'''


# We can examine the correlation between 'engine-size' and 'price' and see that it's approximately 0.87.

print(df[["engine-size", "price"]].corr())


# Highway mpg is a potential predictor variable of price. Let's find the scatterplot of "highway-mpg" and "price".
'''
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

plt.show()


# We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704.

print(df[['highway-mpg', 'price']].corr())


# Weak Linear Relationship

# Let's see if "peak-rpm" is a predictor variable of "price".

sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


print(df[['peak-rpm', 'price']].corr())

plt.show()

# Task: Find the correlation between x="stroke" and y="price".

#The correlation is 0.0823, the non-diagonal elements of the table.

print(df[["stroke","price"]].corr())

# Task: Given the correlation results between "price" and "stroke", do you expect a linear relationship?

#There is a weak correlation between the variable 'stroke' and 'price.' as such regression will not work well. We can see this using "regplot" to demonstrate this.

sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)


plt.show()


# Categorical Variables - These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. 
# The categorical variables can have the type "object" or "int64". A good way to visualize categorical variables is by using boxplots.
# Let's look at the relationship between "body-style" and "price".

sns.boxplot(x="body-style", y="price", data=df)
plt.ylim(0,)

plt.show()


# Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.

sns.boxplot(x="engine-location", y="price", data=df)
plt.ylim(0,)

plt.show()


# Let's examine "drive-wheels" and "price".

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.ylim(0,)

plt.show()

'''

# Descriptive Statistical Analysis  ---  We can apply the method "describe" as follows:

df.describe()

# The default setting of "describe" skips variables of type object. We can apply the method "describe" on the variables of type 'object' as follows:

df.describe(include=['object'])

# Value counts is a good way of understanding how many units of each characteristic/variable we have. We can apply the "value_counts" method on the column "drive-wheels".

df['drive-wheels'].value_counts()

# We can convert the series to a dataframe as follows:

df['drive-wheels'].value_counts().to_frame()

# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column 'drive-wheels' to 'value_counts'.

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)

# Now let's rename the index to 'drive-wheels':

drive_wheels_counts.index.name = 'drive-wheels'

# We can repeat the above process for the variable 'engine-location'.

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))

# After examining the value counts of the engine location, we see that engine location would not be a good predictor variable for the price. 
# This is because we only have three cars with a rear engine and 198 with an engine in the front, so this result is skewed. 
# Thus, we are not able to draw any conclusions about the engine location.

# Basics of Grouping - The "groupby" method groups data by different categories. The data is grouped based on one or several variables, and analysis is performed on the individual groups.
#For example, let's group by the variable "drive-wheels". We see that there are 3 different categories of drive wheels.

print(df['drive-wheels'].unique())


# If we want to know, on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them.
# We can select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".

df_group_one = df[['drive-wheels','body-style','price']]

# We can then calculate the average price for each of the different categories of data.

# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)

# From our data, it seems rear-wheel drive vehicles are, on average, the most expensive, while 4-wheel and front-wheel are approximately the same in price.
# You can also group by multiple variables. For example, let's group by both 'drive-wheels' and 'body-style'. 
# This groups the dataframe by the unique combination of 'drive-wheels' and 'body-style'. We can store the results in the variable 'grouped_test1'.

# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)


# This grouped data is much easier to visualize when it is made into a pivot table. 
# A pivot table is like an Excel spreadsheet, with one variable along the column and another along the row. 
# We can convert the dataframe to a pivot table using the method "pivot" to create a pivot table from the groups.
# In this case, we will leave the drive-wheels variable as the rows of the table, and pivot body-style to become the columns of the table:


grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
print(grouped_pivot)


# Often, we won't have data for some of the pivot cells. We can fill these missing cells with the value 0, but any other value could potentially be used as well. 
# It should be mentioned that missing data is quite a complex subject and is an entire course on its own.


grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
print(grouped_pivot)

# Task: Use the "groupby" function to find the average "price" of each car based on "body-style".

# grouping results
df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
print(grouped_test_bodystyle)


# Variables: Drive Wheels and Body Style vs. Price
# Let's use a heat map to visualize the relationship between Body Style vs Price.



#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

# The heatmap plots the target variable (price) proportional to colour with respect to the variables 'drive-wheel' and 'body-style' on the vertical and horizontal axis, respectively. 
# This allows us to visualize how the price is related to 'drive-wheel' and 'body-style'.

# The default labels convey no useful information to us. Let's change that:

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# Correlation and Causation
#Correlation: a measure of the extent of interdependence between variables.
#Causation: the relationship between cause and effect between two variables.
#It is important to know the difference between these two. Correlation does not imply causation. 
#Determining correlation is much simpler the determining causation as causation may require independent experimentation.

# Pearson Correlation is the default method of the function "corr". Like before, we can calculate the Pearson Correlation of the of the 'int64' or 'float64' variables.

df.corr()

# Wheel-Base vs. Price  --  Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# Conclusion:
# Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585).


# Horsepower vs. Price  Let's calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'.

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

# Conclusion:
# Since the p-value is < 0.001, the correlation between horsepower and price is statistically significant, and the linear relationship is quite strong (~0.809, close to 1).



# Length vs. Price  Let's calculate the Pearson Correlation Coefficient and P-value of 'length' and 'price'.


pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)


# Conclusion:
# Since the p-value is < 0.001, the correlation between length and price is statistically significant, and the linear relationship is moderately strong (~0.691).


#Width vs. Price   Let's calculate the Pearson Correlation Coefficient and P-value of 'width' and 'price':



pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 

# Conclusion:
#Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear relationship is quite strong (~0.751).


# Curb-Weight vs. Price
# Let's calculate the Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price'

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

# Conclusion:
#Since the p-value is < 0.001, the correlation between curb-weight and price is statistically significant, and the linear relationship is quite strong (~0.834).



# Engine-Size vs. Price
# Let's calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price':


pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# Conclusion:
#Since the p-value is < 0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).


# Bore vs. Price
# Let's calculate the Pearson Correlation Coefficient and P-value of 'bore' and 'price':


pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )


# Conclusion:
# Since the p-value is < 0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521).



# City-mpg vs. Price


pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# Conclusion:
#Since the p-value is < 0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of about -0.687 shows that the relationship is negative and moderately strong.



# Highway-mpg vs. Price


pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 


#Conclusion:
#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of about -0.705 shows that the relationship is negative and moderately strong.

