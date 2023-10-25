import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"


# Load the dataset to a pandas dataframe named 'df'   Print the first 5 entries of the dataset to confirm loading.

df = pd.read_csv(url, header=0)
print(df.head(5))


# Verify loading by displaying the dataframe summary using dataframe.info()

print(df.info())

# We can update the Screen_Size_cm column such that all values are rounded to nearest 2 decimal places by using numpy.round()

df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']],2)

# Evaluate the dataset for missing data

missing_data = df.isnull()
print(missing_data.head())
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

# replacing missing data with mean
avg_weight=df['Weight_kg'].astype('float').mean(axis=0)
df["Weight_kg"].replace(np.nan, avg_weight, inplace=True)

# astype() function converts the values to the desired data type
# axis=0 indicates that the mean value is to calculated across all column elements in a row.

# Missing values in attributes that have categorical data are best replaced using the most frequent value. 
# We note that values in "Screen_Size_cm" attribute are categorical in nature, and some values are missing. 
# Therefore, write a code to replace the missing values of Screen Size with the most frequent value of the attribute.

# replacing missing data with mode
common_screen_size = df['Screen_Size_cm'].value_counts().idxmax()
df["Screen_Size_cm"].replace(np.nan, common_screen_size, inplace=True)

# Fixing the data types

df[["Weight_kg","Screen_Size_cm"]] = df[["Weight_kg","Screen_Size_cm"]].astype("float")

# Data Standardization - The value of Screen_size usually has a standard unit of inches. Similarly, weight of the laptop is needed to be in pounds. 
# Use the below mentioned units of conversion and write a code to modify the columns of the dataframe accordingly. Update their names as well. 
# 1 inch = 2.54 cm           1 kg   = 2.205 pounds

# Data standardization: convert weight from kg to pounds
df["Weight_kg"] = df["Weight_kg"]*2.205
df.rename(columns={'Weight_kg':'Weight_pounds'}, inplace=True)

# Data standardization: convert screen size from cm to inch
df["Screen_Size_cm"] = df["Screen_Size_cm"]*2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'}, inplace=True)

# Data Normalization - Often it is required to normalize a continuous data attribute. Write a code to normalize the "CPU_frequency" attribute with respect to the maximum value available in the dataset.

df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].max()

# Binning is a process of creating a categorical attribute which splits the values of a continuous data into a specified number of groups. 
# In this case, write a code to create 3 bins for the attribute "Price". These bins would be named "Low", "Medium" and "High". The new attribute will be named "Price-binned".
# Also, plot the bar graph of these bins.


bins = np.linspace(min(df["Price"]), max(df["Price"]), 4)
group_names = ['Low', 'Medium', 'High']
df['Price-binned'] = pd.cut(df['Price'], bins, labels=group_names, include_lowest=True )


plt.bar(group_names, df["Price-binned"].value_counts())
plt.xlabel("Price")
plt.ylabel("count")
plt.title("Price bins")



# draw historgram of attribute "price" with bins = 3
plt.hist(df["Price"], bins = 3)
plt.xlabel("Price")
plt.ylabel("count")
plt.title("Price bins")


# Display the histogram  <<---------------- show histogram ------------------------<<
plt.show()



# Indicator variables - Convert the "Screen" attribute of the dataset into 2 indicator variables, "Screen-IPS_panel" and "Screen-Full_HD". Then drop the "Screen" attribute from the dataset.


#Indicator Variable: Screen
dummy_variable_1 = pd.get_dummies(df["Screen"])
dummy_variable_1.rename(columns={'IPS Panel':'Screen-IPS_panel', 'Full HD':'Screen-Full_HD'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "Screen" from "df"
df.drop("Screen", axis = 1, inplace=True)

print(df.head())


