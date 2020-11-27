import pathlib
import os

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import constants

import dash_table
import matplotlib.pyplot as plt
import plotly.express as px
from googletrans import Translator
import plotly.offline as pyo
import plotly.graph_objs as go
import json

demographics = pd.read_csv('data\demographics.csv')
employment = pd.read_csv('data\employment.csv')
mobility = pd.read_csv('data\mobility.csv')
cases = pd.read_csv(
    'https://raw.githubusercontent.com/J535D165/CoronaWatchNL/master/data/rivm_NL_covid19_total_municipality.csv')
income = pd.read_csv('data\income.csv')
coordinates = pd.read_csv('data\coordinates.csv')
clustering = pd.read_csv('data\H5cluster.csv')

# Translating the columns
translator = Translator()
cases.rename(columns=lambda x: translator.translate(x).text, inplace=True)

# Removing commas and changing the data types
for i in range(1, demographics.shape[1]):
    if (demographics.iloc[:, i]).dtype == 'object':
        demographics.iloc[:, i] = demographics.iloc[:, i].str.replace(',', '.')
        demographics.iloc[:, i] = demographics.iloc[:, i].astype('float')

for i in range(1, mobility.shape[1]):
    if (mobility.iloc[:, i]).dtype == 'object':
        mobility.iloc[:, i] = mobility.iloc[:, i].str.replace(',', '.')
        mobility.iloc[:, i] = mobility.iloc[:, i].astype('float')

income = income.dropna(axis=1)
for i in range(1, income.shape[1]):
    if (income.iloc[:, i]).dtype == 'object':
        income.iloc[:, i] = income.iloc[:, i].replace('?', '0')
        income.iloc[:, i] = income.iloc[:, i].str.replace(',', '.')
        income.iloc[:, i] = income.iloc[:, i].astype('float')

# Calculating the percentages
demographics.iloc[:, np.r_[4:20]] = demographics.iloc[:, np.r_[4:20]].div(demographics.iloc[:, 1], axis=0)
demographics.iloc[:, np.r_[26:28]] = demographics.iloc[:, np.r_[26:28]].div(demographics.iloc[:, 25], axis=0)
demographics.iloc[:, np.r_[24:26]] = demographics.iloc[:, np.r_[24:26]].div(demographics.iloc[:, 23], axis=0)

# Aggregating the total cases per municipality
cases = cases.dropna()
cases = cases.rename(columns={'municipality Name': 'Municipalities'})
cases = cases.replace("'s-Gravenhage", 'Den Haag')
cases = cases.replace('Nuenen, Gerwen en Nederwetten', 'Nuenen c.a.')

grp1 = cases[cases.date == max(cases.date)]
grp1 = grp1.iloc[:, np.r_[2, 5]]

# coordinates
coordinates = coordinates.rename(
    columns={'Municipality': 'Municipalities', 'Latitude (generated)': 'lat', 'Longitude (generated)': 'lon'})
coordinates = coordinates.apply(lambda x: x.str.replace(',', '.'))
coordinates[['lat', 'lon']] = coordinates[['lat', 'lon']].apply(pd.to_numeric)

# correct misspellings to match
coordinates.Municipalities = coordinates.Municipalities.str.replace("'s-Gravenhage", 'Den Haag')
coordinates.Municipalities = coordinates.Municipalities.str.replace('Nuenen, Gerwen en Nederwetten', 'Nuenen c.a.')

# Geojson with NL data
with open('data\gemeente.geojson') as json_data:
    nl_data = json.load(json_data)

# Join data with municipality id codes (to work with the geojson)
# Read data from website
mun_codes = pd.read_excel('https://www.cbs.nl/-/media/_excel/2020/03/gemeenten-alfabetisch-2020.xlsx')

# Select columns of interest
mun_codes = mun_codes.filter(['GemeentecodeGM', 'Gemeentenaam'], axis=1)
# Rename columns
mun_codes.columns = ['id', 'Municipalities']

# joining the data with the clusters
joined_data = pd.merge(grp1, mun_codes, how='left', on='Municipalities')
joined_data = pd.merge(joined_data, coordinates, how='left', on='Municipalities')
joined_data = pd.merge(joined_data, clustering, how='left', on='Municipalities')
joined_data.Cluster = joined_data.Cluster.astype(str)

# NL latitude and longitude values
latitude = 52.370216
longitude = 4.895168

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

missedData = pd.read_csv('data\Missed-data.csv')


# New case function
def newCase(num):
    num_dropna = num.dropna()
    num_dropna.reset_index(inplace=True)
    num_dropna = num_dropna[['date', 'Municipalities', 'Provincienaam', 'Number']]
    new_cases = pd.DataFrame(
        data={'date': [], 'Municipalities': [], 'Provincienaam': [], 'Number': [], 'New_cases': []})
    Municipality = num_dropna['Municipalities'].unique()
    for i in Municipality:
        M = num_dropna[num_dropna['Municipalities'] == i]
        M.sort_values(by='date', inplace=True)
        M.reset_index(inplace=True)
        M = M[['date', 'Municipalities', 'Provincienaam', 'Number']]
        row_1st = pd.DataFrame(data={'date': [M.iloc[0]['date']], 'Municipalities': [M.iloc[0]['Municipalities']],
                                     'Provincienaam': [M.iloc[0]['Provincienaam']], 'Number': [M.iloc[0]['Number']],
                                     'New_cases': [M.iloc[0]['Number']]})
        new = M['Number'].diff().tolist()
        M.drop([0], inplace=True)
        new.pop(0)
        M['New_cases'] = new
        dif = pd.concat([row_1st, M], ignore_index=True)
        new_cases = pd.concat([new_cases, dif], ignore_index=True)
    return new_cases


missedData = missedData.replace("'s-Gravenhage", 'Den Haag')
missedData = missedData.replace('Nuenen, Gerwen en Nederwetten', 'Nuenen c.a.')

# Adding the missed data to the new cases
new_cases = newCase(cases)
new_cases.set_index('date', inplace=True)
new_cases.drop('2020-04-08', inplace=True)
new_cases.reset_index(level=0, inplace=True)
final_data = pd.concat([new_cases, missedData], ignore_index=True)
final_data['date'] = pd.to_datetime(final_data['date'])
final_data.sort_values(by='date', inplace=True)

# Hierarchical clustering results
dataHierarchical = pd.merge(final_data, clustering, how='left', on='Municipalities')
dataHierarchical.fillna(0, inplace=True)
dataHierarchical['date'] = pd.to_datetime(dataHierarchical['date'])
dataHierarchical.Cluster = dataHierarchical.Cluster.astype(str)

pca_variables = pd.read_csv('data\pca_variables.csv')
pca_variables = pd.merge(pca_variables,grp1,on='Municipalities')
variables = pca_variables[[ 'Number', 'demography_0', 'demography_1', 'demography_2',
       'demography_3', 'employment', 'mobility_0', 'mobility_1', 'mobility_2',
       'mobility_3', 'mobility_4', 'mobility_5', 'mobility_6', 'mobility_7',
       'mobility_8', 'ChronicDisease_0', 'ChronicDisease_1',
       'Avg. PI. per person by household position']]

def randomForest(features):
    X = features.iloc[:, 1:]
    y = features.iloc[:, 0]
    rf = RandomForestRegressor()
    rf.fit(X, y)
    names = X.columns
    top_features = rf.feature_importances_[rf.feature_importances_ > 0.05]
    names = names[rf.feature_importances_ > 0.05]
    return top_features, names

top_features, names = randomForest(variables)

# Map plot function
def plot(data):
    fig = px.choropleth_mapbox(data, geojson=nl_data, locations='id',color='Cluster',
                               mapbox_style="carto-positron",
                               zoom=6, center={"lat": latitude, "lon": longitude},
                               opacity=0.6,
                               labels={'Number': 'Corona Cases'},
                               hover_name='Municipalities',
                               hover_data=['Number'],
                               # color_discrete_map = {'1':'#061324' ,'2':'#D9372B','3':' #3B658F','4':'#F27222','5':'#F28927'}

                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, coloraxis_showscale=False)
    return fig


def lineplot(data):
    fig = px.line(data, x="date", y='New_cases', line_group='Municipalities',
                  hover_name='Municipalities', color='Cluster', color_discrete_map = {'1':'#083554' ,'2':'#D9372B','3':' #3B658F','4':'#F27222','5':'#F28927'})
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin={"r": 20, "t": 20, "l": 0, "b": 0})
    return fig

# Bar plot
def barplot():
    fig = px.bar(x=top_features, y=names, orientation='h')
    fig.update_layout(yaxis={'categoryorder': 'total descending'}, autosize=False,
                      xaxis_title="Score", yaxis_title="Features", showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
    return fig

# app initialize
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
app.config["suppress_callback_exceptions"] = True


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.H6("COVID-19"),
        ],
    )


def build_graph_title(title):
    return html.P(className="graph-title", children=title)



app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children="Select Clusters and you will see the the municipalities in different clusters, and ...",
                                ),
                                build_graph_title("Select cluster"),
                                dcc.Dropdown(id='cluster',
                                             options=[{'label': 'All', 'value': 'All'},
                                                      {'label': '1', 'value': '1'},
                                                      {'label': '2', 'value': '2'},
                                                      {'label': '3', 'value': '3'},
                                                      {'label': '4', 'value': '4'},
                                                      {'label': '5', 'value': '5'}], value='All',
                                             )

                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
                        # Well map
                        html.Div(
                            id="well-map-container",
                            children=[
                                build_graph_title("Netherlands"),
                                html.Label('Cases'),
                                dcc.RangeSlider(id='slider',
                                    min=0,
                                    max=500,
                                    step=1,
                                    marks={
                                        0: '0',
                                        100: '100',
                                        200: '200',
                                        300: '300',
                                        400: '400',
                                        500: '>500'
                                    },
                                    value=[0, 500]
                                                ),
                                dcc.Loading(
                                    # id = 'loading',
                                    children = dcc.Graph(
                                        id="map",
                                        figure={
                                            "layout": {
                                                "paper_bgcolor": "#192444",
                                                "plot_bgcolor": "#192444",
                                            }
                                        },
                                        config={"scrollZoom": True, "displayModeBar": True},
                                    ),
                                ),
                            ],style={'width': '49%','display': 'inline-block','vertical-align': 'middle'}
                        ),
                        # Ternary map
                        html.Div(
                            id="ternary-map-container",
                            children=[
                                html.Div(
                                    id="ternary-header",
                                    children=[
                                        build_graph_title(
                                            "Trend lines"
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children = [dcc.Loading(
                                        # id='loading',
                                        children = [dcc.Graph(
                                            id="trend",
                                            figure={
                                                "layout": {
                                                    "paper_bgcolor": "#192444",
                                                    "plot_bgcolor": "#192444",
                                                }
                                            },
                                            # config={
                                            #     "scrollZoom": True,
                                            #     "displayModeBar": False,
                                            # },
                                        )]
                                    ),
                                ],style={'margin':'7rem 0 0 5rem'})
                            ],style={'width': '49%', 'display': 'inline-block','vertical-align': 'middle'}
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="bottom-row",
            children=[
                # Formation bar plots
                html.Div(
                    id="form-bar-container",
                    className="six columns",
                    children=[
                        build_graph_title("Important Features"),
                        dcc.Loading(
                            id= 'loading',
                            children = dcc.Graph(id="bar-chart"),
                        ),
                    ],style = {'fontColor':'#1f2c56'}
                ),
        #         html.Div(
        #             # Selected well productions
        #             id="well-production-container",
        #             className="six columns",
        #             children=[
        #                 build_graph_title("Individual well annual production"),
        #                 dcc.Loading(
        #                     id= 'loading',
        #                     children = dcc.Graph(id="production-fig"),
        #                 )
        #             ],
        #         ),
            ],
        ),
    ]
)


# Update bar plot
@app.callback(
    Output("bar-chart", "figure"),
    [
        Input('cluster', 'value'),
    ],
)
def update_bar(values):
    fig = barplot()
    return fig



# Update map
@app.callback(
    Output('map', 'figure'),
    [Input('cluster', 'value'),
    Input('slider', 'value')])
def map(cluster, values):
    if cluster == 'All':
        df = joined_data
        if max(values) == 500:
            df = df[(df['Number'] >= min(values))]
        else:
            df = df[(df['Number'] >= min(values)) & (df['Number'] <= max(values))]
        fig = plot(df)
        return fig
    else:
        df = joined_data[joined_data.Cluster == str(cluster)]
        if max(values) == 500:
            df = df[(df['Number'] >= min(values))]
        else:
            df = df[(df['Number'] >= min(values)) & (df['Number'] <= max(values))]
        fig = plot(df)
        return fig
@app.callback(
    Output("trend", 'figure'),
    [Input('cluster', 'value')])
def trend(cluster):
    if cluster == 'All':
        df = dataHierarchical
        fig = lineplot(df)
        return fig
    else:
        df = dataHierarchical[dataHierarchical.Cluster == str(cluster)]
        fig = lineplot(df)
        return fig


# Update production plot
# @app.callback(
#     Output("production-fig", "figure"),
#     [
#         Input("map", "selectedData"),
#         Input("trend", "selectedData"),
#         Input("bar-chart", "selectedData"),
#         Input("operator-select", "value"),
#     ],
# )
# def update_production(map_select, tern_select, bar_select, op_select):
#     dff = df[df["op"].isin(op_select)]
#
#     # Find which one has been triggered
#     ctx = dash.callback_context
#
#     prop_id = ""
#     prop_type = ""
#     if ctx.triggered:
#         splitted = ctx.triggered[0]["prop_id"].split(".")
#         prop_id = splitted[0]
#         prop_type = splitted[1]
#
#     processed_data_init = {}
#     processed_data_init["well_id"] = dff["RecordNumber"].tolist()
#     processed_data_init["formation"] = dff["fm_name"].tolist()
#
#     if prop_id == "map" and prop_type == "selectedData":
#         if map_select is not None:
#             processed_data = {"well_id": [], "formation": []}
#             for point in map_select["points"]:
#                 processed_data["well_id"].append(point["customdata"])
#                 processed_data["formation"].append(
#                     dff[dff["RecordNumber"] == point["customdata"]]["fm_name"].tolist()[
#                         0
#                     ]
#                 )
#         else:
#             processed_data = processed_data_init
#
#     elif prop_id == "trend" and prop_type == "selectedData":
#         if tern_select is not None:
#             processed_data = {"well_id": [], "formation": []}
#             for point in tern_select["points"]:
#                 if "customdata" in point:
#                     processed_data["well_id"].append(point["customdata"])
#                     processed_data["formation"].append(
#                         dff[dff["RecordNumber"] == point["customdata"]][
#                             "fm_name"
#                         ].tolist()[0]
#                     )
#
#         else:
#             processed_data = processed_data_init
#
#     elif prop_id == "bar-chart" and prop_type == "selectedData":
#         if bar_select is not None:
#             processed_data = {"well_id": [], "formation": []}
#
#             # Find all wells according to selected formation category
#             for point in bar_select["points"]:
#                 selected_form = point["x"]
#                 selected_well = dff[dff["fm_name"] == point["x"]][
#                     "RecordNumber"
#                 ].tolist()
#                 for well in selected_well:
#                     processed_data["well_id"].append(int(well))
#                     processed_data["formation"].append(selected_form)
#
#         else:
#             processed_data = processed_data_init
#     else:
#         processed_data = processed_data_init
#
#     return generate_production_plot(processed_data)
#

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
