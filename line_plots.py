# Objectives
# Create Data Visualization with Python
# Use various Python libraries for visualization

import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library

# Let's download and import our primary Canadian Immigration dataset using pandas's read_csv () method.

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')
print('Data read into a pandas dataframe!')

# Let's view the top 5 rows of the dataset using the head() function.

print(df_can.head(4))

# tip: You can specify the number of rows you'd like to see as follows: df_can.head(10) 
# Let's set Country as the index, it will help you to plot the charts easily, by refering to the country names as index value

df_can.set_index('Country', inplace=True)
# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()
#let's check

print(df_can.head(3))

# optional: to remove the name of the index

df_can.index.name = None

# Since we converted the years to string, let's declare a variable that will allow us to easily call upon the full range of years:

# useful for plotting later on
years = list(map(str, range(1980, 2014)))
print(years)


# Visualizing Data using Matplotlib

import matplotlib as mpl
import matplotlib.pyplot as plt

# apply a style to Matplotlib.

print(plt.style.available)
mpl.style.use(['ggplot']) # optional: for ggplot-like style


# Plotting in pandas

# Plot a line graph of immigration from Haiti using df.plot(). First, we will extract the data series for Haiti.

#Since we converted the years to string, 
#let's declare a variable that will allow us to easily call upon the full range of years:

years = list(map(str, range(1980, 2014)))

#creating data series

haiti = df_can.loc['Haiti', years] # passing in years 1980 - 2013 to exclude the 'total' column
haiti.head()

# Next, we will plot a line plot by appending .plot() to the haiti dataframe.

haiti.plot()

# Also, let's label the x and y axis using plt.title(), plt.ylabel(), and plt.xlabel() as follows:

haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

plt.show() # need this line to show the updates made to the figure

# We can clearly notice how number of immigrants from Haiti spiked up from 2010 as Canada stepped up its efforts to accept refugees from Haiti. 
# Let's annotate this spike in the plot by using the plt.text() method.

# However, notice that years are of type string. Let's change the type of the index values to integer first.


haiti.index = haiti.index.map(int) 
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

# annotate the 2010 Earthquake. 
# syntax: plt.text(x, y, label)
plt.text(2000, 6000, '2010 Earthquake') # see note below

plt.show() 


# Let's compare the number of immigrants from India and China from 1980 to 2013.

# Step 1: Get the data set for China and India, and display the dataframe.

df_CI = df_can.loc[['India', 'China'], years]
print(df_CI)

# Step 2: Plot graph. We will explicitly specify line plot by passing in kind parameter to plot().

df_CI.plot(kind='line')

# That doesn't look right. Recall that pandas plots the indices on the x-axis and the columns as individual lines on the y-axis. 
# Since df_CI is a dataframe with the country as the index and years as the columns, we must first transpose the dataframe using transpose() method to swap the row and columns.



df_CI = df_CI.transpose()
df_CI.head()

# pandas will auomatically graph the two countries on the same graph. Go ahead and plot the new transposed dataframe. Make sure to add a title to the plot and label the axes

df_CI.index = df_CI.index.map(int) # let's change the index values of df_CI to type integer for plotting
df_CI.plot(kind='line')

plt.title('Immigrants from China and India')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


# That's because haiti is a series as opposed to a dataframe, and has the years as its indices as shown below.

print(type(haiti))
print(haiti.head(5))


# Task: Compare the trend of top 5 countries that contributed the most to immigration to Canada.


#Step 1: Get the dataset. Recall that we created a Total column that calculates cumulative immigration by country. 
#We will sort on this column to get our top 5 countries using pandas sort_values() method.

inplace = True # paramemter saves the changes to the original df_can dataframe
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)

# get the top 5 entries
df_top5 = df_can.head(5)

# transpose the dataframe
df_top5 = df_top5[years].transpose() 

print(df_top5)


#Step 2: Plot the dataframe. To make the plot more readeable, we will change the size using the `figsize` parameter.
df_top5.index = df_top5.index.map(int) # let's change the index values of df_top5 to type integer for plotting
df_top5.plot(kind='line', figsize=(14, 8)) # pass a tuple (x, y) size



plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


'''
Other Plots
There are many other plotting styles available other than the default Line plot, all of which can be accessed by passing kind keyword to plot(). 
The full list of available plots are as follows:

bar for vertical bar plots
barh for horizontal bar plots
hist for histogram
box for boxplot
kde or density for density plots
area for area plots
pie for pie plots
scatter for scatter plots
hexbin for hexbin plot

'''