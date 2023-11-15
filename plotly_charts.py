# Basic Plotly Charts

# Get Started with Different Chart types in Plotly

# Import required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Scatter Plot:

# A scatter plot shows the relationship between 2 variables on the x and y-axis. 
# The data points here appear scattered when plotted on a two-dimensional plane. 
# Using scatter plots, we can create exciting visualizations to express various relationships, such as:
# - Height vs weight of persons
# - Engine size vs automobile price
# - Exercise time vs Body Fat

##Example 1: Let us illustrate the income vs age of people in a scatter plot

age_array=np.random.randint(25,55,60)
# Define an array containing salesamount values 
income_array=np.random.randint(300000,700000,3000000)

##First we will create an empty figure using go.Figure()
fig=go.Figure()

#Data type check
print(type(fig))

#Next we will create a scatter plot by using the add_trace function and use the go.scatter() function within it
# In go.Scatter we define the x-axis data,y-axis data and define the mode as markers with color of the marker as blue
fig.add_trace(go.Scatter(x=age_array, y=income_array, mode='markers', marker=dict(color='blue')))

# However in the previous output title, x-axis and y-axis labels are missing. Let us use the update_layout function to update the title and labels.

## Here we update these values under function attributes such as title,xaxis_title and yaxis_title
fig.update_layout(title='Economic Survey', xaxis_title='Age', yaxis_title='Income')
# Display the figure
fig.show()

# Inferences:
# From the above plot we find that the Income of a person is not correlated with age. We find that as the age increases the income may or not decrease.

# 2. Line Plot:
# A line plot shows information that changes continuously with time. Here the data points are connected by straight lines. 
# Line plots are also plotted on a two dimensional plane like scatter plots. Using line plots, we can create exciting visualizations to illustrate:
# - Annual revenue growth
# - Stock Market analysis over time
# - Product Sales over time


##Example 2: Let us illustrate the sales of bicycles from Jan to August last year using a line chart
# Define an array containing numberofbicyclessold  
numberofbicyclessold_array=[50,100,40,150,160,70,60,45]
# Define an array containing months
months_array=["Jan","Feb","Mar","April","May","June","July","August"]

##First we will create an empty figure using go.Figure()
fig=go.Figure()
#Next we will create a line plot by using the add_trace function and use the go.scatter() function within it
# In go.Scatter we define the x-axis data,y-axis data and define the mode as lines with color of the marker as green
fig.add_trace(go.Scatter(x=months_array, y=numberofbicyclessold_array, mode='lines', marker=dict(color='green')))
## Here we update these values under function attributes such as title,xaxis_title and yaxis_title
fig.update_layout(title='Bicycle Sales', xaxis_title='Months', yaxis_title='Number of Bicycles Sold')
# Display the figure
fig.show()

# Inferences: From the above plot we find that the sales is the highest in the month of May and then there is a decline in sales.
# We will now use plotly express library to plot the other graphs

# 3.Bar Plot:
# A bar plot represents categorical data in rectangular bars. Each category is defined on one axis, and the value counts for this category are represented on another axis. 
# Bar charts are generally used to compare values.We can use bar plots in visualizing:
# - Pizza delivery time in peak and non peak hours
# - Population comparison by gender
# - Number of views by movie name

##Example 3: Let us illustrate the average pass percentage of classes from grade 6 to grade 10

# Define an array containing scores of students 
score_array=[80,90,56,88,95]
# Define an array containing Grade names  
grade_array=['Grade 6','Grade 7','Grade 8','Grade 9','Grade 10']

# In plotly express we set the axis values and the title within the same function call 
# px.<graphtype>(x=<xaxis value source>,y=<y-axis value source>,title=<appropriate title as a string>).
# In the below code we use px.bar( x=grade_array, y=score_array, title='Pass Percentage of Classes').


# Use plotly express bar chart function px.bar.Provide input data, x and y axis variable, and title of the chart.
# This will give average pass percentage per class
fig = px.bar( x=grade_array, y=score_array, title='Pass Percentage of Classes') 
fig.show()

# From the above plot we find that Grade 8 has the lowest pass percentage and Grade 10 has the highest pass percentage

# 4.Histogram:
# A histogram is used to represent continuous data in the form of bar. 
# Each bar has discrete values in bar graphs, whereas in histograms, we have bars representing a range of values. 
# Histograms show frequency distributions. We can use histograms to visualize:
# - Students marks distribution
# - Frequency of waiting time of customers in a Bank


##Example 4: Let us illustrate the distribution of heights of 200 people using a histogram

import numpy as np
#Here we will concentrate on heights which are 160 and the standard deviation is 11
heights_array = np.random.normal(160, 11, 200)
## Use plotly express histogram chart function px.histogram.Provide input data x to the histogram
fig = px.histogram(x=heights_array,title="Distribution of Heights")
fig.show()

# From this we can analyze that there are around 2 people who are at the height of 130cm and 45 people at the height of 160 cm

# 5. Bubble Plot:
# A bubble plot is used to show the relationship between 3 or more variables. It is an extension of a scatter plot. Bubble plots are ideal for visualizing:
# - Global Economic position of Industries
# - Impact of viruses on Diseases

##Example 4: Let us illustrate crime statistics of US cities with a bubble chart

#Create a dictionary having city,numberofcrimes and year as 3 keys
crime_details = {
    'City' : ['Chicago', 'Chicago', 'Austin', 'Austin','Seattle','Seattle'],
    'Numberofcrimes' : [1000, 1200, 400, 700,350,1500],
    'Year' : ['2007', '2008', '2007', '2008','2007','2008'],
}
  
# create a Dataframe object with the dictionary
df = pd.DataFrame(crime_details)
  
df

## Group the number of crimes by city and find the total number of crimes per city
bub_data = df.groupby('City')['Numberofcrimes'].sum().reset_index()

##Display the grouped dataframe
bub_data

## Bubble chart using px.scatter function with x ,y and size varibles defined.Title defined as Crime Statistics
fig = px.scatter(bub_data, x="City", y="Numberofcrimes", size="Numberofcrimes",
                 hover_name="City", title='Crime Statistics', size_max=60)
fig.show()

# The size of the bubble in the bubble chart indicates that Chicago has the highest crime rate when compared with the other 2 cities.

# 6.Pie Plot:
# A pie plot is a circle chart mainly used to represent proportion of part of given data with respect to the whole data. 
# Each slice represents a proportion and on total of the proportion becomes a whole. We can use bar plots in visualizing:
# - Sales turnover percentatge with respect to different products
# - Monthly expenditure of a Family

## Monthly expenditure of a family
# Random Data
exp_percent= [20, 50, 10,8,12]
house_holdcategories = ['Grocery', 'Rent', 'School Fees','Transport','Savings']

# Use px.pie function to create the chart. Input dataset. 
# Values parameter will set values associated to the sector. 'exp_percent' feature is passed to it.
# labels for the sector are passed to the `house hold categoris` parameter.
fig = px.pie(values=exp_percent, names=house_holdcategories, title='Household Expenditure')
fig.show()

# From this pie chart we can find that the family expenditure is maximum for rent.

# 7.Sunburst Charts:
# Sunburst charts represent hierarchial data in the form of concentric circles. 
# Here the innermost circle is the root node which defines the parent, and then the outer rings move down the hierarchy from the centre. 
# They are also called radial charts.We can use them to plot

# Worldwide mobile Sales where we can drill down as follows:

# - innermost circle represents total sales
# -first outer circle represents continentwise sales
# -second outer circle represents countrywise sales within each continent
# Disease outbreak hierarchy
# Real Estate Industrial chain

##Example 4: Let us illustrate plot the 

#Create a dictionary having a set of people represented by a character array and the parents of these characters represented in another
## array and the values are the values associated to the vectors.
data = dict(
    character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parent=["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
    value=[10, 14, 12, 10, 2, 6, 6, 4, 4])

fig = px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
    title="Family chart"
)
fig.show()

# It is found that here the innermost circle Eve represents the parent and the second outer circle represents his childrent Cain,Seth and so on.
# Further the outermost circle represents his grandchildren Enoch and Enos

# Practice Exercises: Apply your Plotly Skills to an Airline Dataset
# The Reporting Carrier On-Time Performance Dataset contains information on approximately 200 million domestic US flights reported to the United States Bureau of 
# Transportation Statistics. The dataset contains basic information about each flight (such as date, time, departure airport, arrival airport) and, if applicable, 
# the amount of time the flight was delayed and information about the reason for the delay. This dataset can be used to predict the likelihood of a flight arriving on time.
