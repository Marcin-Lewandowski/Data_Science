# Accessing Databases with SQLite3
'''

Introduction
Using this Python notebook you will:

Understand three Chicago datasets
Load the three datasets into three tables in a SQLIte database
Execute SQL queries to answer assignment questions


To complete the assignment problems in this notebook you will be using three datasets that are available on the city of Chicago's Data Portal:

Socioeconomic Indicators in Chicago
Chicago Public Schools
Chicago Crime Data
'''

import pandas as pd, sqlite3

# Import CSV files into pandas dataframes
# Chicago Census Data

url_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCensusData.csv?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDB0201ENSkillsNetwork20127838-2021-01-01'
df_1 = pd.read_csv(url_1)

# Chicago Public Schools

url_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoPublicSchools.csv?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDB0201ENSkillsNetwork20127838-2021-01-01'
df_2 = pd.read_csv(url_2)

# Chicago Crime Data

url_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCrimeData.csv?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDB0201ENSkillsNetwork20127838-2021-01-01'
df_3 = pd.read_csv(url_3)


# Create a connection to the SQLite database
conn = sqlite3.connect('FinalDB.db')

cur = conn.cursor()

# Write the dataframes to tables in the SQLite database

df_1.to_sql('CENSUS_DATA', conn, if_exists='replace', index=False)
df_2.to_sql('CHICAGO_PUBLIC_SCHOOLS', conn, if_exists='replace', index=False)
df_3.to_sql('CHICAGO_CRIME_DATA', conn, if_exists='replace', index=False)


# 1. Find the total number of crimes recorded in the CRIME table.
cur.execute('SELECT COUNT(*) FROM CHICAGO_CRIME_DATA')

# Fetch the result of the SELECT statement
result = cur.fetchone()
print()
print('Number of crimes recorded in the table CHICAGO_CRIME_DATA: {}'.format(result[0]))
print()


# 2. List community area names and numbers with per capita income less than 11000

cur.execute('SELECT COMMUNITY_AREA_NAME, COMMUNITY_AREA_NUMBER FROM CENSUS_DATA WHERE PER_CAPITA_INCOME < 11000')

# Fetch the results of the SELECT statement
results = cur.fetchall()

# Print the results
for row in results:
    print(row[0], row[1])


# 3. List all case numbers for crimes involving a child?
cur.execute('SELECT CASE_NUMBER FROM CHICAGO_CRIME_DATA WHERE DESCRIPTION LIKE "%CHILD%"')

# Fetch the results of the SELECT statement
results = cur.fetchall()
print()

# Print the results
for row in results:
    print(row[0])



# 4. List all kidnapping crimes involving a child
cur.execute('SELECT CASE_NUMBER FROM CHICAGO_CRIME_DATA WHERE PRIMARY_TYPE="KIDNAPPING" AND DESCRIPTION LIKE "%CHILD%"')

# Fetch the results of the SELECT statement
results = cur.fetchall()
print()

# Print the results
for row in results:
    print(row[0])



# 5. List the kind of crimes that were recorded at schools. (No repetitions)

cur.execute('SELECT DISTINCT PRIMARY_TYPE FROM CHICAGO_CRIME_DATA WHERE LOCATION_DESCRIPTION LIKE "%SCHOOL%"')

# Fetch the results of the SELECT statement
results = cur.fetchall()
print()

# Print the results
for row in results:
    print(row[0])



# 6. List the type of schools along with the average safety score for each type.

cur.execute('SELECT \"Elementary, Middle, or High School\", AVG(Safety_Score) FROM \"CHICAGO_PUBLIC_SCHOOLS\" GROUP BY \"Elementary, Middle, or High School\"')

# Fetch the results of the SELECT statement
results = cur.fetchall()
print()

# Print the results
for row in results:
    print(row[0], row[1])


# 7. List 5 community areas with highest % of households below poverty line

cur.execute('SELECT COMMUNITY_AREA_NAME FROM CENSUS_DATA ORDER BY PERCENT_HOUSEHOLDS_BELOW_POVERTY DESC LIMIT 5')

# Fetch the results of the SELECT statement
results = cur.fetchall()
print()

# Print the results
for row in results:
    print(row[0])



# 8. Which community area is most crime prone? Display the coumminty area number only.

cur.execute('SELECT COMMUNITY_AREA_NUMBER FROM CHICAGO_CRIME_DATA GROUP BY COMMUNITY_AREA_NUMBER ORDER BY COUNT(*) DESC LIMIT 1')

# Fetch the result of the SELECT statement
result = cur.fetchone()
print()

# Print the result
print('The community area number with the highest crime rate is: {}'.format(result[0]))


# 9. Use a sub-query to find the name of the community area with highest hardship index

cur.execute('SELECT COMMUNITY_AREA_NAME FROM CENSUS_DATA WHERE HARDSHIP_INDEX = (SELECT MAX(HARDSHIP_INDEX) FROM CENSUS_DATA)')

# Fetch the result of the SELECT statement
result = cur.fetchone()
print()

# Print the result
print('The community area with the highest hardship index is: {}'.format(result[0]))


# 10. Use a sub-query to determine the Community Area Name with most number of crimes?

cur.execute('SELECT COMMUNITY_AREA_NAME FROM CENSUS_DATA WHERE COMMUNITY_AREA_NUMBER = (SELECT COMMUNITY_AREA_NUMBER FROM CHICAGO_CRIME_DATA GROUP BY COMMUNITY_AREA_NUMBER ORDER BY COUNT(*) DESC LIMIT 1)')

# Fetch the result of the SELECT statement
result = cur.fetchone()
print()

# Print the result
print('The community area with the most number of crimes is: {}'.format(result[0]))



# Close the connection to the SQLite database
conn.close()