# Data Vizualization Practice Project part 1

# Part 1 : Analyzing the wildfire activities in Australia
# Part 2 : Dashboard to display charts based on selected Region and Year


'''
Data Description
This wildfire dataset contains data on fire activities in Australia starting from 2005. Additional information can be found here.
The dataset includes the following variables:

Region: the 7 regions
Date: in UTC and provide the data for 24 hours ahead
Estimated_fire_area: daily sum of estimated fire area for presumed vegetation fires with a confidence > 75% for a each region in km2
Mean_estimated_fire_brightness: daily mean (by flagged fire pixels(=count)) of estimated fire brightness for presumed vegetation fires with a confidence level > 75% in Kelvin
Mean_estimated_fire_radiative_power: daily mean of estimated radiative power for presumed vegetation fires with a confidence level > 75% for a given region in megawatts
Mean_confidence: daily mean of confidence for presumed vegetation fires with a confidence level > 75%
Std_confidence: standard deviation of estimated fire radiative power in megawatts
Var_confidence: Variance of estimated fire radiative power in megawatts
Count: daily numbers of pixels for presumed vegetation fires with a confidence level of larger than 75% for a given region
Replaced: Indicates with an Y whether the data has been replaced with standard quality data when they are available (usually with a 2-3 month lag). 
Replaced data has a slightly higher quality in terms of locations
'''

# Part 1 : Analyzing the wildfire activities in Australia
# Objective:
# The objective of this part of the Practice Assignment is to analyze and visualize the wildfire activities in Australia using the provided dataset. 
# You will explore patterns and trends, and create visualizations to gain insights into the behavior of wildfires in different regions of Australia.
# In this lab you will create visualizations using Matplotlib, Seaborn, Pandas and Folium.

'''
# Tasks to be performed:
TASK 1.1: To understand the change in average estimated fire area over time using pandas to plot the line chart.

TASK 1.2 To plot the estimated fire area over month

TASK 1.3 Use the functionality of seaborn to develop a barplot, to find the insights on the distribution of mean estimated fire brightness across the regions

TASK 1.4 Develop a pie chart and find the portion of count of pixels for presumed vegetation fires vary across regions

TASK 1.5 Customize the previous pie plot for a better visual representation

TASK 1.6 Use Matplotlib to develop a histogram of the mean estimated fire brightness

TASK 1.7 Use the functionality of seaborn and pass region as hue, to understand the distribution of estimated fire brightness across regions

TASK 1.8 Develop a scatter plot to find the correlation between mean estimated fire radiative power and mean confidence level

TASK 1.9 Mark all seven regions affected by wildfires, on the Map of Australia using Folium

'''

# Practice Assignment - Part 1: Analyzing wildfire activities in Australia

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Historical_Wildfires.csv"
df = pd.read_csv(url)
print('Data read into a pandas dataframe!')
print()
print(df.head(5))
print()
print(df.tail(5))
print()
print(df.shape)
print()

# Let's verify the column names and the data type of each variable

#Column names
print(df.columns)
#data type
print(df.dtypes)
 
# Notice the type of 'Date' is object, let's convert it to 'datatime' type and also let's extract 'Year' and 'Month' from date and include in the dataframe as separate columns

import datetime as dt

df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month

# Verify the columns again

# TASK 1.1: Let's try to understand the change in average estimated fire area over time (use pandas to plot)

plt.figure(figsize=(12, 6))
df_new=df.groupby('Year')['Estimated_fire_area'].mean()
df_new.plot(x=df_new.index, y=df_new.values)
plt.xlabel('Year')
plt.ylabel('Average Estimated Fire Area (km²)')
plt.title('Estimated Fire Area over Time')
plt.show()

# TASK 1.2: You can notice the peak in the plot between 2010 to 2013. Let's narrow down our finding, by plotting the estimated fire area for year grouped together with month.
# You will be required to group the year and month for 'Estimated_fire_area' (taking its mean)  Then use df.plot() on it to create a line graph. OK let's try ;)

df_new=df.groupby(['Year','Month'])['Estimated_fire_area'].mean()
df_new.plot(x=df_new.index, y=df_new.values)
plt.xlabel('Year, Month')
plt.ylabel('Average Estimated Fire Area (km²)')
plt.title('Estimated Fire Area over Time')
plt.show()

# This plot represents that the estimated fire area was on its peak after 2011, April and before 2012. 
# You can verify on google/news, this was the time of maximum wildfire hit in Austrailia

# TASK 1.3: Let's have an insight on the distribution of mean estimated fire brightness across the regions - use the functionality of seaborn to develop a barplot
# Make use of unique() to identify the regions in the dataset (apply it on series only)

df['Region'].unique()

# Hint - you need to plot reions on x-axis and the 'Mean_estimated_fire_brightness' on y-axis. Title it as 'Distribution of Mean Estimated Fire Brightness across Regions'

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Region', y='Mean_estimated_fire_brightness')
plt.xlabel('Region')
plt.ylabel('Mean Estimated Fire Brightness (Kelvin)')
plt.title('Distribution of Mean Estimated Fire Brightness across Regions')
plt.show()

# TASK 1.4: Let's find the portion of count of pixels for presumed vegetation fires vary across regions we will develop a pie chart for this
# First you will be required to group the data on region and find the sum of count

plt.figure(figsize=(10, 6))
region_counts = df.groupby('Region')['Count'].sum()
plt.pie(region_counts, labels=region_counts.index)
plt.title('Percentage of Pixels for Presumed Vegetation Fires by Region')
plt.legend([(i,round(k/region_counts.sum()*100,2)) for i,k in zip(region_counts.index, region_counts)])
plt.axis('equal')
plt.show()

# TASK 1.5: See the percentage on the pie is not looking so good as it is overlaped for Region SA, TA, VI
# I need to remove the autopct fromm pie function and pass the following to plt.legend() after plt.title(): DONE  ;))


# TASK 1.6: Let's try to develop a histogram of the mean estimated fire brightness. Using Matplotlib to create the histogram
# Hint: Call plt.hist() and pass df['Mean_estimated_fire_brightness'] as x

plt.figure(figsize=(10, 6))
plt.hist(x=df['Mean_estimated_fire_brightness'], bins=20)
plt.xlabel('Mean Estimated Fire Brightness (Kelvin)')
plt.ylabel('Count')
plt.title('Histogram of Mean Estimated Fire Brightness')
plt.show()

# TASK 1.7: What if we need to understand the distribution of estimated fire brightness across regions? Let's use the functionality of seaborn and pass region as hue

sns.histplot(data=df, x='Mean_estimated_fire_brightness', hue='Region')
plt.xlabel('Mean Estimated Fire Brightness (Kelvin)')
plt.ylabel('Count')
plt.title('Histogram of Mean Estimated Fire Brightness')
plt.show()

# Looks better! Now include the parameter multiple='stack' in the histplot() and see the difference. Include labels and titles as well

sns.histplot(data=df, x='Mean_estimated_fire_brightness', hue='Region', multiple='stack')
plt.xlabel('Mean Estimated Fire Brightness (Kelvin)')
plt.ylabel('Count')
plt.title('Histogram of Mean Estimated Fire Brightness')
plt.show()


# TASK 1.8: Let's try to find if there is any correlation between mean estimated fire radiative power and mean confidence level?
# Call plt.scatter() or use the --->> sns.scatterplot()  <<---

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Mean_confidence', y='Mean_estimated_fire_radiative_power')
plt.xlabel('Mean Estimated Fire Radiative Power (MW)')
plt.ylabel('Mean Confidence')
plt.title('Mean Estimated Fire Radiative Power vs. Mean Confidence')
plt.show()

# TASK 1.9: Let's mark these seven regions on the Map of Australia using Folium

# Dataframe for you containing the regions, their latitudes and longitudes. For australia use [-25, 135] as location to create the map

region_data = {
    'region': ['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA', 'NT'],
    'Lat': [-31.8759835, -22.1646782, -30.5343665, -42.035067, -37.8136, -25.2303005, -19.491411],
    'Lon': [147.2869493, 144.5844903, 135.6301212, 146.6366887, 144.9631, 121.0187246, 132.550964]
}
reg=pd.DataFrame(region_data)
reg

# instantiate a feature group 
aus_reg = folium.map.FeatureGroup()

# Create a Folium map centered on Australia
Aus_map = folium.Map(location=[-25, 135], zoom_start=4)

# loop through the region and add to feature group
for lat, lng, lab in zip(reg.Lat, reg.Lon, reg.region):
    aus_reg.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            popup=lab,
            radius=5, # define how big you want the circle markers to be
            color='red',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add incidents to map
Aus_map.add_child(aus_reg)

map_filename = "c://kodilla/Data_Science/Data_Vizualization/australian_regions_map.html"
Aus_map.save(map_filename)