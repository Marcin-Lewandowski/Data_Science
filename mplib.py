# Plotting directly with Matplotlib

import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt

# check for latest version of Matplotlib
print('Matplotlib version: ', mpl.__version__) 

# Fetching Data
# Dataset: Immigration to Canada from 1980 to 2013 - International migration flows to and from selected countries - The 2015 revision from United Nation's website.
# In this lab, we will focus on the Canadian Immigration data and use the already cleaned dataset and can be fetched from here.

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv"
df_can = pd.read_csv(url)

print('Data read into a pandas dataframe!')
print(df_can.head(4))

# print the dimensions of the dataframe
print(df_can.shape)

# Set the country name as index - useful for quickly looking up countries using .loc method.

df_can.set_index('Country', inplace=True)

# Let's view the first five elements and see how the dataframe was changed
print(df_can.head(4))

# Notice now the country names now serve as indices.

print('data dimensions:', df_can.shape)

# finally, let's create a list of years from 1980 - 2013
# this will come in handy when we start plotting the data

years = list(map(str, range(1980, 2014)))

#years = np.arange(1980,2014)

'''
Line Plot
A line plot displays the relationship between two continuous variables over a continuous interval, showing the trend or pattern of the data.

Let's created a line plot to visualize the immigrants (to Canada) trend during 1980 to 2013.
We need the Total of year-wise immigrants,

We will create a new dataframe for only columns containing the years
then, we will apply sum() on the dataframe

You can do create a line plot directly on axes by calling plot function plot()
'''

#As years is in the array format, you will be required to map it to str for plotting
#y=list(map(str, years))

#creating df with only years columns from 1980 - 2013
df_line=df_can[years]

#Applying sum to get total immigrants year-wise
total_immigrants=df_line.sum()
total_immigrants

#Create figure and axes
fig, ax = plt.subplots()

# Plot the line
ax.plot(total_immigrants)

#Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 
#Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')

#Display the plot
plt.show()

# The plot function populated the x-axis with the index values (years), and the y-axis with the column values (population).
# However, notice how the years were not displayed because they are of type string.
# Therefore, let's change the type of the index values to integer for plotting.

#Create figure and axes
fig, ax = plt.subplots()

#Changing the index type to integer
total_immigrants.index = total_immigrants.index.map(int)

# Plot the line
ax.plot(total_immigrants)

#Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 

#Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')

#Display the plot
plt.show()


# Let's now customize the above plot's appearance
#Create figure and axes
fig, ax = plt.subplots()

#Changing the index type to integer
total_immigrants.index = total_immigrants.index.map(int)

# Customizing the appearance of Plot
ax.plot(total_immigrants, 
        marker='s', #Including markers in squares shapes
        markersize=5, #Setting the size of the marker
        color='green', #Changing the color of the line
        linestyle="dotted") #Changing the line style to a Dotted line
#Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 

#Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')
ax.legend(['Immigrants'])

plt.show()

# Let's include the background grid, a legend and try to change the limits on the axis

#Create figure and axes
fig, ax = plt.subplots()

# Plot the line
ax.plot(total_immigrants, 
        marker='s', #Including markers in squares shapes
        markersize=5, #Setting the size of the marker
        color='green', #Changing the color of the line
        linestyle="dotted") #Changing the line style to a Dotted line

#Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 

#Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')

#limits on x-axis
plt.xlim(1975, 2015)  #or ax.set_xlim()

#Enabling Grid
plt.grid(True)  #or ax.grid()

#Legend
plt.legend(["Immigrants"]) #or ax.legend()

#Display the plot
plt.show()


# Let's start with a case study:
# In 2010, Haiti suffered a catastrophic magnitude 7.0 earthquake. The quake caused widespread devastation and loss of life and aout three million people were affected by this natural disaster. 
# As part of Canada's humanitarian effort, the Government of Canada stepped up its effort in accepting refugees from Haiti. We can quickly visualize this effort using a Line plot:

# Task - Plot a line graph of immigration from Haiti

# You be required to create a dataframe where the name of the 'Country' is equal to 'Haiti' and years from 1980 - 2013
# Also you will be required to transpose the new dataframe in to a series for plotting
# Might also have to change the type of index of the series to integer for a better look of the plot
# Then create fig and ax and call function plot() on the data.

#Creating data for plotting
df_can.reset_index(inplace=True)
haiti=df_can[df_can['Country']=='Haiti']

#creating haiti with only years columns from 1980 - 2013 
#and transposing to get the result as a series
haiti=haiti[years].T

#converting the index to type integer
haiti.index = haiti.index.map(int)

#Plotting the line plot on the data
fig, ax = plt.subplots()
ax.plot(haiti)
#Setting up the Title
ax.set_title('Immigrants from Haiti between 1980 to 2013') 
#Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Number of Immigrants')
#Enabling Grid
#plt.grid(True)  #or ax.grid()
#Legend
plt.legend(["Immigrants"]) #or ax.legend()
#Display the plot
plt.show()


# We can clearly notice how number of immigrants from Haiti spiked up from 2010 as Canada stepped up its efforts to accept refugees from Haiti.
# Let's annotate this spike in the plot by using the ax.annotate() method.

fig, ax = plt.subplots()

ax.plot(haiti)

#Setting up the Title
ax.set_title('Immigrants from Haiti between 1980 to 2013') 

#Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Number of Immigrants')

#Enabling Grid and ticks
#plt.grid(True)  #or ax.grid()
#ax.set_xticks(list(range(n, m, s)))

#Legend
plt.legend(["Immigrants"]) #or ax.legend()

ax.annotate('2010 Earthquake',xy=(2000, 6000))
plt.show()


# Scatter Plot
'''
A scatter plot visualizes the relationship between two continuous variables, displaying individual data points as dots on a two-dimensional plane, 
allowing for the examination of patterns, clusters, and correlations.

Let's created a Scatter plot to visualize the immigrants (to Canada) trend during 1980 to 2013.
We need the Total of year-wise immigrants,

We will create a new dataframe for only columns containing the years
then, we will apply sum() on the dataframe

You can do create a scatter plot directly on ax by calling plot function scatter()
'''

#Create figure and axes
fig, ax = plt.subplots(figsize=(8, 4))

# Customizing Scatter Plot 
ax.scatter(years, total_immigrants, 
           marker='o', #setting up the markers
           s = 20, #setting up the size of the markers
           color='darkblue')#the color for the marker

#add title 
plt.title('Immigrants between 1980 to 2013') 
#add labels 
plt.xlabel('Years')
plt.ylabel('Total Immigrants') 
#including grid
plt.grid(True)

#Legend at upper center of the figure
ax.legend(["Immigrants"], loc='upper center')

#Display the plot
plt.show()


#Create figure and axes
fig, ax = plt.subplots(figsize=(8, 4))

total_immigrants.index = total_immigrants.index.map(int)

# Customizing Scatter Plot 
ax.scatter(total_immigrants.index, total_immigrants, 
           marker='o', #setting up the markers
           s = 20, #setting up the size of the markers
           color='darkblue')#the color for the marker

#add title 
plt.title('Immigrants between 1980 to 2013') 
#add labels 
plt.xlabel('Years')
plt.ylabel('Total Immigrants') 
#including grid
plt.grid(True)

#Legend at upper center of the figure
ax.legend(["Immigrants"], loc='upper center')

#Display the plot
plt.show()



# Bar Plot
'''
A bar plot represents categorical data with rectangular bars, where the height of each bar corresponds to the value of a specific category, making it suitable for comparing values across different categories.

Let's create a bar plot to visualize the top 5 countries that contribued the most immigrants to Canada from 1980 to 2013.

Apply sort_values function on the 'Total' column of our data

We will create a new dataframe for only columns containing the years
then, we will apply sum() on the dataframe and can create a separatedataframe for top five countries

You can further use the names of the countries to label each bar on the plot
'''

#Sorting the dataframe on 'Total' in descending order
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

# get the top 5 entries with head function
df_top5 = df_can.head()

#resetting the index back to original way
df_bar_5=df_top5.reset_index()

#Creating alist of names of the top 5 countries
label=list(df_bar_5.Country)
label

# The third name is too lengthy to fit on the x-axis as label. Let's fix this using indexing

label[2]='UK'
print(label)

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(label,df_bar_5['Total'], label=label)
ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

plt.show()

# Create a bar plot of the 5 countries that contributed the least to immigration to Canada from 1980 to 2013

#Sorting the dataframe on 'Total' in descending order
df_can.sort_values(['Total'], ascending=True, axis=0, inplace=True)

# get the top 5 entries with head function
df_least5 = df_can.head()

#resetting the index back to original way
df_least5_bar=df_least5.reset_index()

#Creating alist of names of the top 5 countries
label=list(df_least5_bar.Country)
#label

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(label, df_least5_bar['Total'],label=label)
ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

plt.show()


# Histogram
'''
A histogram is a way of representing the frequency distribution of numeric dataset. 
The way it works is it partitions the x-axis into bins, assigns each data point in our dataset to a bin, and then counts the number of data points that have been assigned to each bin. 
So the y-axis is the frequency or the number of data points in each bin. Note that we can change the bin size and usually one needs to tweak it so that the distribution is displayed nicely.

Let's find out the frequency distribution of the number (population) of new immigrants from the various countries to Canada in 2013?
'''

df_country = df_can.groupby(['Country'])['2013'].sum().reset_index()

#Create figure and axes
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_country['2013'])
ax.set_title('New Immigrants in 2013') 
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Number of Countries')
ax.legend(['Immigrants'])

#Display the plot
plt.show()

# Our plot doesnot match with the bars. By default, the histrogram method breaks up the dataset into 10 bins. 
# The figure below summarizes the bin ranges and the frequency distribution of immigration in 2013
# The hist function retuns list of arrays with 1. counts and 2. bins. we can fetch that using unpacking functionality and further use the bins as x-ticks

# Plot the bar
fig, ax = plt.subplots(figsize=(10, 4))
count = ax.hist(df_country['2013'])

#you can check the arrays in count with indexing count[0] for count, count[1] for bins

ax.set_title('New Immigrants in 2013') 
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Number of Countries')
ax.set_xticks(list(map(int,count[1])))
ax.legend(['Immigrants'])

#Display the plot
plt.show()

# We can also plot multiple histograms on the same plot. For example, let's try to answer the following questions using a histogram.
# What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?

# let's quickly view the dataset 
df=df_can.groupby(['Country'])[years].sum()
df_dns=df.loc[['Denmark', 'Norway', 'Sweden'], years]
df_dns=df_dns.T
df_dns

#Create figure and axes
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_dns)
ax.set_title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013') 
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Number of Years')
ax.legend(['Denmark', 'Norway', 'Sweden'])
#Display the plot
plt.show()

# Task - What is the immigration distribution for China and India for years 2000 to 2013?

# let's quickly view the dataset 
df=df_can.groupby(['Country'])[years].sum()
y=list(map(str,range(2000, 2014)))
df_ci=df.loc[['China', 'India'], y]
df_ci=df_ci.T
#df_ci
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_ci)
ax.set_title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013') 
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Number of Years')
ax.legend(['China', 'India'])
#Display the plot
plt.show()


# Pie Chart - A pie chart represents the proportion or percentage distribution of different categories in a dataset using sectors of a circular pie.
# Let's create a pie chart representing the 'Total Immigrants' for the year 1980 to 1985

fig,ax=plt.subplots()

#Pie on immigrants
ax.pie(total_immigrants[0:5], labels=years[0:5], 
       colors = ['gold','blue','lightgreen','coral','cyan'],
       autopct='%1.1f%%',explode = [0,0,0,0,0.1]) #using explode to highlight the lowest 

ax.set_aspect('equal')  # Ensure pie is drawn as a circle

plt.title('Distribution of Immigrants from 1980 to 1985')
#plt.legend(years[0:5]), include legend, if you donot want to pass the labels
plt.show()

# Task - Create a pie chart representing the total immigrants proportion for each continent

# First, you will have to group the data over continents and get the sum on total. Then you can pass this data to the pie function

#Creating data for plotting pie
df_con=df_can.groupby('Continent')['Total'].sum().reset_index()
label=list(df_con.Continent)
label[3] = 'LAC'
label[4] = 'NA'
print(df_con)

fig,ax=plt.subplots(figsize=(10, 4))

#Pie on immigrants
ax.pie(df_con['Total'], colors = ['gold','blue','lightgreen','coral','cyan','red'],
        autopct='%1.1f%%', pctdistance=1.25)

ax.set_aspect('equal')  # Ensure pie is drawn as a circle

plt.title('Continent-wise distribution of immigrants')
ax.legend(label,bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

# Sub-plotting

# Let us explore how to display more than one plot on the same figure and specify the number of rows and columns to be created to the subplots function.

# For instance, let’s create a line and scatter plot in one row plt.subplots()
# You can use the same functions using which you plotte lne and scatter plots at the start of this lab
# Both the subplots will be sharing the same y-axis as the data in the y-axis is the same. So, assign the ‘Sharey’ parameter as True in the code below. 
# Also notice the use of 'suptitle'

# Create a figure with two axes in a row

fig, axs = plt.subplots(1, 2, sharey=True)

#Plotting in first axes - the left one
axs[0].plot(total_immigrants)
axs[0].set_title("Line plot on immigrants")

#Plotting in second axes - the right one
axs[1].scatter(total_immigrants.index, total_immigrants)
axs[1].set_title("Scatter plot on immigrants")

axs[0].set_ylabel("Number of Immigrants")
            
#Adding a Title for the Overall Figure
fig.suptitle('Subplotting Example', fontsize=15)

# Adjust spacing between subplots
fig.tight_layout()


# Show the figure
plt.show()

# You can also implement the subplotting with add_subplot() as below:

# Create a figure with Four axes - two rows, two columns
fig = plt.figure(figsize=(8,4))

# Add the first subplot (top-left)
axs1 = fig.add_subplot(1, 2, 1)
#Plotting in first axes - the left one
axs1.plot(total_immigrants)
axs1.set_title("Line plot on immigrants")

# Add the second subplot (top-right)
axs2 = fig.add_subplot(1, 2, 2)
#Plotting in second axes - the right one
axs2.barh(total_immigrants.index, total_immigrants) #Notice the use of 'barh' for creating horizontal bar plot
axs2.set_title("Bar plot on immigrants")
            
#Adding a Title for the Overall Figure
fig.suptitle('Subplotting Example', fontsize=15)

# Adjust spacing between subplots
fig.tight_layout()


# Show the figure
plt.show()

# Task - Choose any four plots, which you have developed in this lab, with subplotting display them in a 2x2 display

# Create a figure with Four axes - two rows, two columns
fig = plt.figure(figsize=(10, 10))

# Add the first subplot (top-left)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(total_immigrants)
ax1.set_title('Plot 1 - Line Plot')

# Add the second subplot (top-right)
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(total_immigrants.index, total_immigrants)
ax2.set_title('Plot 2 - Scatter plot')

# Add the third subplot (bottom-left)
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(df_dns)
ax3.set_title('Plot3 - Histogram') 
ax3.set_xlabel('Number of Immigrants')
ax3.set_ylabel('Number of Years')

# Add the fourth subplot (bottom-right)
ax4 = fig.add_subplot(2, 2, 4)
ax4.pie(total_immigrants[0:5], labels=years[0:5], 
        colors = ['gold','blue','lightgreen','coral','cyan'],
        autopct='%1.1f%%')
ax4.set_aspect('equal')  
ax4.set_title('Plot 5 - Pie Chart')

#Adding a Title for the Overall Figure
fig.suptitle('Four Plots in a Figure Example', fontsize=15)

# Adjust spacing between subplots
fig.tight_layout()


# Show the figure
plt.show()


