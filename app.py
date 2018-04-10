# Import required libraries
import matplotlib
matplotlib.use('Agg') ## Avoid tkinter bug
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
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#
#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),
#
#     dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'title': 'Dash Data Visualization'
#             }
#         }
#     )
# ])



## West-Life

# "PMC4942797" -  Bilayer Membrane Modulation of Membrane Type 1 Matrix Metalloproteinase (MT1-MMP) Structure and Proteolytic Activity - Cerofolini et al.
# "PMC3567692" -  Gentamicin Binds to the Megalin Receptor as a Competitive Inhibitor Using the Common Ligand Binding Motif of Complement Type Repeats: INSIGHT FROM THE NMR STRUCTURE OF THE 10TH COMPLEMENT TYPE REPEAT DOMAIN ALONE AND IN COMPLEX WITH GENTAMICIN - Kragelund et al.
# "PMC5552742" -  Kinetic and Structural Characterization of the Effects of Membrane on the Complex of Cytochrome b 5 and Cytochrome c - Gentry et al.

ext_id_list = ["PMC5552742",
               "PMC4942797",
               # "PMC3567692", no fulltext XML
               ]

ext_id = ext_id_list[0]

r = requests.get("https://github.com/RobFirth/pyresid-heroku-dash-app/raw/master/" + ext_id + ".pkl")
full_matches = pickle.load(io.BytesIO(r.content))

df = pd.DataFrame([match.__dict__ for match in full_matches])

n_items = 2
if len(df.residue.unique()) <= n_items:
    init_threshold = 0
else:
    threshold_count = np.array([[i, len(df.residue.unique()[df.residue.value_counts() >= i])] for i in
                                range(df.residue.value_counts().max())]).T
    print(threshold_count)
    init_threshold = np.where(threshold_count[1] <= n_items)[0][0]
print(init_threshold)
    # w_init_threshold = np.where(np.array([len(df.residue.value_counts()[df.residue.value_counts() > i]) for i in df.residue.value_counts().unique()[::-1]])>=n_items)[0][-1]
    # init_threshold = [len(df.residue.value_counts()[df.residue.value_counts() > i]) for i in df.residue.value_counts().unique()[::-1]][w_init_threshold]
# residues_to_plot = df.residue.value_counts()[:n_items].index.tolist()


app = dash.Dash()
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

# text_style = dict(color='#444', fontFamily='OpenSans', fontWeight=300)
text_style = dict()

app.layout = html.Div(children=[
    html.Div(children=[
        html.Div(html.Img(src="https://raw.githubusercontent.com/RobFirth/RobFirth.github.io/master/images/pyresid_logo_whitebackground.png",
                          width="100%"), className="three columns"),
        html.Div(className="one column"),
        html.Div(children=[html.H1('Protein Residue Dashboard'),
                           dcc.Markdown("An Amino-Acid Residue identifier and tagger "
                                  "using `python`. Feed me fulltextXML "
                                  "of interesting papers!")], className="eight columns"),
        ], className="row", id="title-row"
        ),
    html.Div(children=[
        html.Div(html.Img(
            src="https://raw.githubusercontent.com/RobFirth/RobFirth.github.io/master/images/CombinedLogo.png",
            width="100%"), className="twelve columns"),
    ], className="row", id="logo-row"),
    html.Hr(),
    html.Div(children=[
        html.P(
            "A dashboard using pyresid.")]),
    html.Hr(),
    ## The Actual Business End!
    html.Div(children=[
        dcc.Dropdown(
            id="ext_id-select-input",
            options=[{'label': i, 'value': i} for i in list(ext_id_list)
                     ],
            value=ext_id_list[0]
            ),
    html.Div(id="ext_id-output-div")
    ], className="row", id="ext_id-row"),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'}),

    # html.Div(children=[
    # ], className="row", id="info-row"),
    # print("foo"),
    html.Div(children=[
    html.H4("Residue Location Plot"),
    dcc.Graph(
        id="locplot-with-slider"
    ),
    html.Div(children=html.H5("Select the threshold number of mentions:")),
    print(init_threshold),
    html.Div(html.Div(dcc.Slider(
        id="residue-slider",
        min = df.residue.value_counts().min(),
        max = df.residue.value_counts().max(),
        # value=df.residue.value_counts().max(),
        value=init_threshold,
        step=None,
        # marks=list(range(df.residue.value_counts().max()))
        marks={str(value) : str(value) for value in df.residue.value_counts().unique()[::-1]}
        ),
        id="residue-slider-div", style = {'textAlign': 'center'}
    )),
    ],style = {'textAlign': 'center'}, className="row", id="locplot-and-slider-row"),
    html.Hr(),
    html.Div(children=[
    html.H3("Annotations Table"),
    html.P("Choose the residue that you would like the context for."),
    html.Div(children=[dcc.Dropdown(
            id="residue_select-input",
            ),
    ],
    id="residue_select-input-div"),
    html.Div(id="table_anno-div")
    ], className="row", id="residue-row"),
    # html.Hr(),
    html.Div(children=[
    html.Footer("2018 - Rob Firth; STFC Hartree Centre; West-Life")
    ], className="row", id="footer-row"),
], className="container")



@app.callback(
    dash.dependencies.Output(component_id="ext_id-output-div", component_property='children'),
    [dash.dependencies.Input(component_id="ext_id-select-input", component_property='value')]
)
def update_output_div(input_ext_id):
    meta = pyre.get_metadata(ext_id=input_ext_id)
    print(meta)
    return ['You\'ve entered {}\n'.format(input_ext_id),
            html.H4(meta["title"]),
            html.P(html.Strong(meta["authors"][0]["surname"] + " et al."))
            ]


@app.callback(dash.dependencies.Output('intermediate-value', 'children'),
    [dash.dependencies.Input(component_id="ext_id-select-input", component_property='value')]
)
def collect_data(value):
    print(value)
    ext_id = value

    r = requests.get("https://github.com/RobFirth/pyresid-heroku-dash-app/raw/master/" + ext_id + ".pkl")
    full_matches = pickle.load(io.BytesIO(r.content))

    df = pd.DataFrame([match.__dict__ for match in full_matches])

    return df.to_json(date_format='iso', orient='split')


@app.callback(
    dash.dependencies.Output('locplot-with-slider', 'figure'),
    [dash.dependencies.Input('intermediate-value', 'children'),
     dash.dependencies.Input('residue-slider', 'value')]
)
def update_figure(stored_data, selected_valuecount):
    df=pd.read_json(stored_data, orient="split")
    filtered_df = df[df["residue"].isin(df.residue.value_counts()[df.residue.value_counts() >= selected_valuecount].index.tolist())]

    points = []


    for i, j in enumerate(filtered_df.residue.value_counts().index):
        print(j)
        df_by_residue = filtered_df[filtered_df["residue"] == j]

        points.append(go.Scatter(
                        x = df_by_residue['token_start'],
                        y = (len(filtered_df.residue.value_counts())-i)*np.ones(len(df_by_residue)),
                        text=df_by_residue['string'],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=j
        ))
    return {
        "data" : points,
        "layout": go.Layout(
            xaxis=dict(title="Token Number", showspikes=True),
            yaxis=dict(title='Residue', tickvals=list(np.arange(len(filtered_df.residue.value_counts().index))+1)[::-1],
                       ticktext=[item[0]+" ("+str(item[1])+")" for item in zip([i for i in filtered_df.residue.value_counts().index], [j for j in filtered_df.residue.value_counts()])],
                       showspikes=True),
            margin={'l': "15%", 'b': "5%", 't': "2%", 'r': "5%"},
            hovermode='closest'
        )
    }


@app.callback(
    dash.dependencies.Output(component_id="residue-slider-div", component_property="children"),
    [dash.dependencies.Input(component_id="intermediate-value", component_property="children")]
)
def update_slider(stored_data):
    df=pd.read_json(stored_data, orient="split")

    n_items = 5
    if len(df.residue.unique()) <= n_items:
        init_threshold = 0
    else:
        threshold_count = np.array([[i, len(df.residue.unique()[df.residue.value_counts() >= i])] for i in
                                    range(df.residue.value_counts().max())]).T
        print(threshold_count)
        init_threshold = np.where(threshold_count[1] <= n_items)[0][0]
    print(init_threshold)

    return html.Div(dcc.Slider(
        id="residue-slider",
        min = df.residue.value_counts().min(),
        max = df.residue.value_counts().max(),
        # value=df.residue.value_counts().max(),
        value=init_threshold,
        step=None,
        # marks=list(range(df.residue.value_counts().max()))
        marks={str(value) : str(value) for value in df.residue.value_counts().unique()[::-1]}
        )
    )


@app.callback(
    dash.dependencies.Output(component_id="residue_select-input-div", component_property='children'),
    [dash.dependencies.Input(component_id='intermediate-value', component_property='children')]
)
def update_dropdown(stored_data):
    df = pd.read_json(stored_data, orient="split")

    return html.Div(dcc.Dropdown(
            id="residue_select-input",
            options=[{'label': i, 'value': i} for i in list(df.residue.unique())
                     ],
            value=df.residue.value_counts().index[0]
            ))

@app.callback(
    dash.dependencies.Output(component_id="table_anno-div", component_property='children'),
    [dash.dependencies.Input('intermediate-value', 'children'),
     dash.dependencies.Input(component_id='residue_select-input', component_property='value')]
)
def update_table(stored_data, input_value):
    df=pd.read_json(stored_data, orient="split")
    return generate_table(df[df["residue"] == input_value])



# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
