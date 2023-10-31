import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline


# House Sales in King County, USA
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.
# In this assignment, you are a Data Analyst working at a Real Estate Investment Trust. The Trust would like to start investing in Residential real estate. 
# You are tasked with determining the market price of a house given a set of features. 
# You will analyze and predict housing prices using attributes or features such as square footage, number of bedrooms, number of floors, and so on.

'''
Variable	        Description
id	                A notation for a house
date	            Date house was sold
price	            Price is prediction target
bedrooms	        Number of bedrooms
bathrooms	        Number of bathrooms
sqft_living	        Square footage of the home
sqft_lot	        Square footage of the lot
floors	            Total floors (levels) in house
waterfront	        House which has a view to a waterfront
view	            Has been viewed
condition	        How good the condition is overall
grade	            overall grade given to the housing unit, based on King County grading system
sqft_above	        Square footage of house apart from basement
sqft_basement	    Square footage of the basement
yr_built	        Built Year
yr_renovated	    Year when house was renovated
zipcode	            Zip code
lat	                Latitude coordinate
long	            Longitude coordinate
sqft_living15	    Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
sqft_lot15	        LotSize area in 2015(implies-- some renovations)


'''

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'


df = pd.read_csv(url)
print(df.head(5))

print(df.dtypes)

# We use the method describe to obtain a statistical summary of the dataframe.

print(df.describe())


# Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. 
# Make sure the inplace parameter is set to True


# Drop the columns "id" and "Unnamed: 0" from the dataset

df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)


# We can see we have missing values for the columns  bedrooms and  bathrooms 

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# We can replace the missing values of the column 'bedrooms' with the mean of the column 'bedrooms'  using the method replace(). Don't forget to set the inplace parameter to True

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

# We also replace the missing values of the column 'bathrooms' with the mean of the column 'bathrooms' using the method replace(). Don't forget to set the inplace parameter to True 

mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# Module 3: Exploratory Data Analysis

# Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.


# Count the number of houses with unique floor values
floor_counts = df['floors'].value_counts().to_frame()

# Display the counts as a dataframe
print(floor_counts)



# Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# Create a boxplot to compare the distribution of prices for houses with and without a waterfront view
plt.figure(figsize=(8, 6))
sns.boxplot(x='waterfront', y='price', data=df)
plt.xlabel('Waterfront View', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Price Distribution for Houses with and without Waterfront View', fontsize=16)


# Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.


# Create a regression plot to assess the correlation between sqft_above and price
plt.figure(figsize=(8, 6))
sns.regplot(x='sqft_above', y='price', data=df, scatter_kws={"s": 10}, line_kws={"color": "red"})
plt.xlabel('Square Footage Above', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Correlation between Square Footage Above and Price', fontsize=16)


# We can use the Pandas method corr() to find the feature other than price that is most correlated with price.

df.corr()['price'].sort_values()


# Model Development
# We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2. Take a screenshot of your code and the value of the R^2.

# Fit a linear regression model using the 'sqft_living' feature and calculate the R^2.
X_sqft_living = df[['sqft_living']]
Y_price = df['price']
lm_sqft_living = LinearRegression()
lm_sqft_living.fit(X_sqft_living, Y_price)
r_squared_sqft_living = lm_sqft_living.score(X_sqft_living, Y_price)
print("R-squared (R^2) for the linear regression model (sqft_living vs. price):", r_squared_sqft_living)


# Fit a linear regression model to predict the 'price' using the list of features:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]

X_features = df[features]
Y_price = df['price']
lm_features = LinearRegression()
lm_features.fit(X_features, Y_price)
r_squared_features = lm_features.score(X_features, Y_price)
print("R-squared (R^2) for the linear regression model (list of features vs. price):", r_squared_features)



'''
Create a list of tuples, the first element in the tuple contains the name of the estimator:

'scale'

'polynomial'

'model'

The second element in the tuple contains the model constructor

StandardScaler()

PolynomialFeatures(include_bias=False)

LinearRegression()

'''


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# Create a pipeline object using the list of tuples
pipeline = Pipeline(Input)

# Fit the pipeline object using the features in the list 'features'
X_features = df[features]
Y_price = df['price']
pipeline.fit(X_features, Y_price)

# Calculate the R-squared (R^2)
r_squared_pipeline = pipeline.score(X_features, Y_price)
print("R-squared (R^2) for the pipeline model (list of features vs. price):", r_squared_pipeline)


# Model Evaluation and Refinement


# We will split the data into training and testing sets:

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.


# Create a Ridge regression object
ridge = Ridge(alpha=0.1)

# Fit the Ridge model using the training data
ridge.fit(x_train, y_train)

# Calculate the R-squared (R^2) using the test data
r_squared_ridge = ridge.score(x_test, y_test)
print("R-squared (R^2) for Ridge regression model:", r_squared_ridge)


# Perform a second order polynomial transform on both the training data and testing data. 
# Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided.

# Create and fit a second-order polynomial transformation for both training and testing data
poly = PolynomialFeatures(degree=2)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create and fit a Ridge regression object
ridge_poly = Ridge(alpha=0.1)

ridge_poly.fit(x_train_poly, y_train)

# Calculate the R-squared (R^2) using the test data
r_squared_ridge_poly = ridge_poly.score(x_test_poly, y_test)
print("R-squared (R^2) for Ridge regression model with second-order polynomial features:", r_squared_ridge_poly)





# Show the plot
plt.show()