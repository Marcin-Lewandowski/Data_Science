# Regression Trees

# In this lab you will learn how to implement regression trees using ScikitLearn. 
# We will show what parameters are important, how to train a regression tree, and finally how to determine our regression trees accuracy.


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"


# Pandas will allow us to create a dataframe of the data so it can be used and manipulated
import pandas as pd

# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor

# Split our data into a training and testing data
from sklearn.model_selection import train_test_split

'''
About the Dataset
Imagine you are a data scientist working for a real estate company that is planning to invest in Boston real estate. You have collected information about various areas of Boston and are tasked with created a model that can predict the median price of houses for that area so it can be used to make offers.

The dataset had information on areas/towns not individual houses, the features are

CRIM: Crime per capita

ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.

INDUS: Proportion of non-retail business acres per town

CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

NOX: Nitric oxides concentration (parts per 10 million)

RM: Average number of rooms per dwelling

AGE: Proportion of owner-occupied units built prior to 1940

DIS: Weighted distances to ﬁve Boston employment centers

RAD: Index of accessibility to radial highways

TAX: Full-value property-tax rate per $10,000

PTRAIO: Pupil-teacher ratio by town

LSTAT: Percent lower status of the population

MEDV: Median value of owner-occupied homes in $1000s
'''


data = pd.read_csv(url)
print(data.head(7))


# Now lets learn about the size of our data, there are 506 rows and 13 columns

print(data.shape)

# Most of the data is valid, there are rows with missing values which we will deal with in pre-processing

data.isna().sum()

# Data Pre-Processing

# First lets drop the rows with missing values because we have enough data in our dataset

data.dropna(inplace=True)

# Now we can see our dataset has no missing values

print(data.isna().sum())

# Lets split the dataset into our features and what we are predicting (target)

X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
X.head()
Y.head()

# Finally lets split our data into a training and testing dataset using train_test_split from sklearn.model_selection


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

'''
Create Regression Tree
Regression Trees are implemented using DecisionTreeRegressor from sklearn.tree

The important parameters of DecisionTreeRegressor are

criterion: {'squared_error', 'absolute_error', 'poisson', 'friedman_mse'} - The function used to measure error

max_depth - The max depth the tree can be

min_samples_split - The minimum number of samples required to split a node

min_samples_leaf - The minimum number of samples that a leaf can contain

max_features: {"auto", "sqrt", "log2"} - The number of feature we examine looking for the best one, used to speed up training

First lets start by creating a DecisionTreeRegressor object, setting the criterion parameter to squared_error for Squared Error
'''

regression_tree = DecisionTreeRegressor(criterion = "squared_error")

# Training
# Now lets train our model using the fit method on the DecisionTreeRegressor object providing our training data

regression_tree.fit(X_train, Y_train)

# Evaluation
# To evaluate our dataset we will use the score method of the DecisionTreeRegressor object providing our testing data, 
# this number is the R^2 value which indicates the coefficient of determination

regression_tree.score(X_test, Y_test)

# We can also find the average error in our testing set which is the average error in median home value prediction

prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)

# Excercise

# Train a regression tree using the criterion squared_error then report its R^2 value and average error

regression_tree = DecisionTreeRegressor(criterion = "squared_error")

regression_tree.fit(X_train, Y_train)

print(regression_tree.score(X_test, Y_test))

prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)

