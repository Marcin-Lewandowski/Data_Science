import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021-06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()

tesla = yf.Ticker("TSLA")

tesla_data = tesla.history(period="max")
tesla_data.reset_index(inplace=True)


# URL of the webpage to download - zbieranie informacji o revenue firmy Tesla
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm"

# Send an HTTP GET request to the URL and get the HTML content
html_data = requests.get(url).text

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(html_data, 'html5lib')





# Find the table containing Tesla revenue data by inspecting the HTML source
table = soup.find_all("tbody")[1]
data = []

# Przechodzimy przez wiersze i kolumny tabeli i wydobywamy dane
for row in table.find_all("tr"):
    columns = row.find_all("td")
    date = columns[0].get_text()
    revenue = columns[1].get_text()
    data.append({'Date': date, 'Revenue': revenue})

# Tworzymy ramkę danych Pandas na podstawie listy słowników
tesla_revenue = pd.DataFrame(data)
tesla_revenue["Revenue"] = tesla_revenue['Revenue'].str.replace(',|\$',"")
tesla_revenue.dropna(inplace=True)

tesla_revenue = tesla_revenue[tesla_revenue['Revenue'] != ""]




# Wyświetlamy pierwsze kilka wierszy ramki danych
print()
print(tesla_revenue.head())
print()
print()

print(tesla_data.head(5))
print()

make_graph(tesla_data, tesla_revenue, "Tesla")



