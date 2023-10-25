# Data cleaning, data wrangling

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# Use the Pandas method read_csv() to load the data from the web address. Set the parameter "names" equal to the Python list "headers".

df = pd.read_csv(url, names = headers)

# To see what the data set looks like, we'll use the head() method.
#print(df.head(5))

# replace "?" to NaN
df.replace("?", np.NaN, inplace = True)
#print(df.head(5))

# Checkin for Missing Data

missing_data = df.isnull()
#print(missing_data.head(5))

# Oblicza ile True i False jest w każdej kolumnie
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 

# Calculate the mean value for the "normalized-losses" column

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

# Replace "NaN" with mean value in "normalized-losses" column

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

# Calculate the mean value for the "bore" column

avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

# Replace "NaN" with the mean value in the "bore" column

df["bore"].replace(np.nan, avg_bore, inplace=True)

#print(df.head(8))

# Based on the example above, replace NaN in "stroke" column with the mean value.

avg_stroke = df['stroke'].astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)

# Calculate the mean value for the "horsepower" column

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

# Replace "NaN" with the mean value in the "horsepower" column

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

# Calculate the mean value for "peak-rpm" column

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

# Replace "NaN" with the mean value in the "peak-rpm" column

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)



# To see which values are present in a particular column, we can use the ".value_counts()" method:

print(df['num-of-doors'].value_counts())

# You can see that four doors is the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:

df['num-of-doors'].value_counts().idxmax()

# The replacement procedure is very similar to what you have seen previously:

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)


# Finally, drop all rows that do not have price data:

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

print()
print(df.head(20))
print()

# Convert data types to proper format

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# Let us list the columns after the conversion

print(df.dtypes)

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
print(df['city-L/100km'])

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)

df['highway-L/100km'] = 235/df["highway-mpg"]


# check your transformed data 
print(df['highway-L/100km'])

# Data Normalization, scaling the variable so the variable values range from 0 to 1

# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max() 

# show the scaled columns
print(df[["length","width","height"]].head(8))

# Example of Binning Data In Pandas - Convert data to correct format:

df["horsepower"]=df["horsepower"].astype(int, copy=True)



bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

group_names = ['Low', 'Medium', 'High']

# Apply the function "cut" to determine what each value of df['horsepower'] belongs to.

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))

# See the number of vehicles in each bin:

print(df["horsepower-binned"].value_counts())

# Plot the distribution of each bin:

plt.bar(group_names, df["horsepower-binned"].value_counts())


# draw historgram of attribute "horsepower" with bins = 3
plt.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# Display the histogram  <<---------------- show histogram ------------------------<<
#plt.show()

# Get the indicator variables and assign it to data frame "dummy_variable_1":

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())

# Change the column names for clarity:

dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
print(dummy_variable_1.head())

# In the data frame, column 'fuel-type' now has values for 'gas' and 'diesel' as 0s and 1s.

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

# Exercise: Create an indicator variable for the column "aspiration"

# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


# Exercise: Merge the new dataframe to the original dataframe, then drop the column 'aspiration'.

# merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)


# Save the new csv:

df.to_csv('clean_df.csv')