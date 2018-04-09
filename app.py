# Import required libraries
import io
import requests
import pickle
import os
from random import randint
import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
import flask
import dash
from dash.dependencies import Input, Output, State, Event
import dash_core_components as dcc
import dash_html_components as html

def generate_table(dataframe, max_rows=10):
    usecolumns_df = ("residue", "EBI_section", "sent")
    usecolumns_names = ("Residue", "Section", "Sentence Context")

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in usecolumns_names])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in usecolumns_df
        ]) for i in range(min(len(dataframe), max_rows))]
    )


# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server)


# Put your Dash code here
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
