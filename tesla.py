import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Define a function to create a two-subplot graph
def make_graph(stock_data, revenue_data, stock):
    # Create a subplot with two rows and one column
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(
        "Historical Share Price", "Historical Revenue"), vertical_spacing=.3)
    
    # Filter stock data up to a specific date
    stock_data_specific = stock_data[stock_data.Date <= '2021-06-14']
    
    # Filter revenue data up to a specific date
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    
    # Add share price data to the first subplot
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True),
                  y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    
    # Add revenue data to the second subplot
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True),
                  y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    
    # Update the x-axis labels and y-axis labels for both subplots
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    
    # Customize the layout of the graph
    fig.update_layout(showlegend=False,
                      height=900,
                      title=stock,
                      xaxis_rangeslider_visible=True)
    
    # Display the graph
    fig.show()

# Create a Tesla Ticker object
tesla = yf.Ticker("TSLA")

# Retrieve historical stock data for Tesla
tesla_data = tesla.history(period="max")
tesla_data.reset_index(inplace=True)

# URL of the webpage to download - collecting information about Tesla's revenue
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm"

# Send an HTTP GET request to the URL and get the HTML content
html_data = requests.get(url).text

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(html_data, 'html5lib')

# Find the table containing Tesla revenue data by inspecting the HTML source
table = soup.find_all("tbody")[1]
data = []

# Iterate through the rows and columns of the table and extract data
for row in table.find_all("tr"):
    columns = row.find_all("td")
    date = columns[0].get_text()
    revenue = columns[1].get_text()
    data.append({'Date': date, 'Revenue': revenue})

# Create a Pandas DataFrame based on the list of dictionaries
tesla_revenue = pd.DataFrame(data)

# Clean the "Revenue" column by removing commas and dollar signs
tesla_revenue["Revenue"] = tesla_revenue['Revenue'].str.replace(',|\$', "")
tesla_revenue.dropna(inplace=True)

# Filter out rows with empty "Revenue" values
tesla_revenue = tesla_revenue[tesla_revenue['Revenue'] != ""]

# Display the first few rows of the revenue data and stock data
print()
print(tesla_revenue.head())
print()
print(tesla_data.head(5))
print()

# Create the graph using the provided data
make_graph(tesla_data, tesla_revenue, "Tesla")