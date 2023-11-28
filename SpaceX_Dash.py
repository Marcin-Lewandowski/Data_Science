# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("c://kodilla/Data_Science/Data_Vizualization/spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='site-dropdown',   # id='id'
                                            options=[
                                                {'label': site, 'value': site} for site in ['All'] + list(spacex_df['Launch Site'].unique())
                                            ],
                                            value='All',
                                            placeholder="Select a Launch Site here",
                                            searchable=True
                                            ),
                                

                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                
                                dcc.RangeSlider(id='payload-slider',
                                                min=0, max=10000, step=250,
                                                marks={0: '0', 1000: '1000'},
                                                value=[min_payload, max_payload]),


                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output

# Function decorator to specify function input and output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value')) 
        
def get_pie_chart(entered_site):
    if entered_site == 'All':
        # If ALL sites are selected, group the data by launch site and count successful missions
        site_success_counts = spacex_df[spacex_df['class'] == 1].groupby('Launch Site').size().reset_index(name='Success Count')

        # Create a pie chart for the total successful launches for each launch site
        fig = px.pie(site_success_counts, values='Success Count', names='Launch Site', title='Total Success Launches for Each Launch Site')
        return fig
    else:
        # If a specific launch site is selected, filter the dataframe for the selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        
        # Render and return a pie chart for the success (class=1) and failed (class=0) count for the selected site
        fig = px.pie(filtered_df, names='class', title=f'Success vs Failed Launches for {entered_site}')
        return fig
        
        

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              [Input(component_id='site-dropdown', component_property='value'),
               Input(component_id='payload-slider', component_property='value')])



def get_scatter_chart(entered_site, payload_range):
    if entered_site.upper() == 'ALL':
        # Jeśli wybrane są WSZYSTKIE lokalizacje, utwórz wykres punktowy dla całej ramki danych
        fig = px.scatter(spacex_df, x='Payload Mass (kg)', y='class',
                         color='Booster Version Category',
                         title='Payload vs. Launch Success (All Sites)')
    else:
        # Jeśli wybrano konkretną lokalizację startu, przefiltruj ramkę danych dla tej lokalizacji
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        
        # Wygeneruj wykres punktowy dla danej lokalizacji startu
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class',
                         color='Booster Version Category',
                         title=f'Payload vs. Launch Success for {entered_site}')
    
    # Aktualizuj układ, aby uwzględniał zakres ładunku z suwaka
    fig.update_layout(
        xaxis=dict(range=[payload_range[0], payload_range[1]]),
    )
    return fig



# Run the app
if __name__ == '__main__':
    app.run_server()

'''
TASK 1: Add a Launch Site Drop-down Input Component
We have four different launch sites and we would like to first see which one has the largest success count. Then,
we would like to select one specific site and check its detailed success rate (class=0 vs. class=1).

TASK 2: Add a callback function to render success-pie-chart based on selected site dropdown
The general idea of this callback function is to get the selected launch site from site-dropdown and render
a pie chart visualizing launch success counts.

Dash callback function is a type of Python function which will be automatically called by
Dash whenever receiving an input component updates, such as a click or dropdown selecting event.

TASK 3: Add a Range Slider to Select Payload
Next, we want to find if variable payload is correlated to mission outcome. From a dashboard point of view, we
want to be able to easily select different payload range and see if we can identify some visual patterns.


TASK 4: Add a callback function to render the success-payload-scatter-chart scatter plot
Next, we want to plot a scatter plot with the x axis to be the payload and the y axis to be the launch outcome (i.e., class column).
As such, we can visually observe how payload may be correlated with mission outcomes for selected site(s).

In addition, we want to color-label the Booster version on each scatter point so that we may
observe mission outcomes with different boosters.



Finding Insights Visually
Now with the dashboard completed, you should be able to use it to analyze SpaceX launch data, and answer the following questions:

Which site has the largest successful launches?
Which site has the highest launch success rate?
Which payload range(s) has the highest launch success rate?
Which payload range(s) has the lowest launch success rate?
Which F9 Booster version (v1.0, v1.1, FT, B4, B5, etc.) has the highest
launch success rate?

'''