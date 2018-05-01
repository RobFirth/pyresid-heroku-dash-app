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
import spacy as spacy
nlp = spacy.load('en_core_web_md')

import pyresid as pyre

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

def generate_highlight_table(dataframe, max_rows=10):
    usecolumns_df = ("residue", "EBI_section", "sent")
    usecolumns_names = ("Residue", "Section", "Sentence Context")
    table_list = []

    header = html.Tr([html.Th(col) for col in usecolumns_names])

    table_list.append(header)

    for i in range(min(len(dataframe), max_rows)):
        row_list = []

        for col in usecolumns_df:

            if col == "sent":
                children = [dataframe.iloc[i]["prefix"], html.Mark(dataframe.iloc[i]["string"]), dataframe.iloc[i]["postfix"]]
                row_list.append(html.Td(html.P(children=children)))
            else:
                row_list.append(html.Td(dataframe.iloc[i][col]))

        table_list.append(html.Tr(row_list))

    return html.Table(table_list)

# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server)
# Dash CSS
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
# Loading screen CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

ext_id = "PMC5740067"

text_dict = pyre.get_text(ext_id)
fulltext = pyre.reconstruct_fulltext(text_dict, tokenise=False)

source = pyre.SourceClass()
source.ext_id = ext_id
source.text_dict = text_dict
source.sections = list(zip([text_dict[i]["title"] for i in text_dict], [text_dict[i]["offset"] for i in text_dict]))
source.fulltext = fulltext
source.doc = nlp(source.fulltext)
source = pyre.add_sections_to_source(source)

meta = pyre.get_metadata(ext_id)

remake=False

r = requests.get("https://github.com/RobFirth/pyresid-heroku-dash-app/raw/master/" + ext_id + ".pkl")
full_matches = pickle.load(io.BytesIO(r.content))

df = pd.DataFrame([match.__dict__ for match in full_matches])

n_items = 5
w_init_threshold = np.where(np.array([len(df.residue.value_counts()[df.residue.value_counts() > i]) for i in df.residue.value_counts().unique()[::-1]])>=n_items)[0][-1]
init_threshold = [len(df.residue.value_counts()[df.residue.value_counts() > i]) for i in df.residue.value_counts().unique()[::-1]][w_init_threshold]

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
        dcc.Markdown(
            """This is a dashboard using `pyresid`. `pyresid` is available via `pip` on PyPi - [https://pypi.org/project/pyresid/](https://pypi.org/project/pyresid/)
            """),
        html.P(
            "Powered by: "),
        html.Div(children=[html.Div(children=[html.P(),], className="two columns"),
                           html.Div(html.Img(src="https://raw.githubusercontent.com/RobFirth/RobFirth.github.io/master/images/pdbe_logo.png", width="100%"), className="four columns"),
                           html.Div(html.Img(src="https://raw.githubusercontent.com/RobFirth/RobFirth.github.io/master/images/spacy_logo.png", width="100%"), className="four columns"),
                           html.Div(children=[html.P(),], className="two columns"),
                          ], className="row")
        ]),
    html.Hr(),
    ## The Actual Business End!
    html.Div(children=[
    dcc.Input(
        id="ext_id-input",
        placeholder='Enter an ePMC id',
        type='text',
        value=ext_id),
    html.Button(id='ext_id-submit-button', children='Extract'),
    html.Div(id="ext_id-output-div")
    ], className="row", id="ext_id-row"),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'}),

    html.Div(children=[
    html.H4("Residue Location Plot"),
    dcc.Graph(
        id="locplot-with-slider"
    ),
    html.Div(children=html.H5("Select the threshold number of mentions:")),
    html.Div(html.Div(dcc.Slider(
        id="residue-slider",
        min = df.residue.value_counts().min(),
        max = df.residue.value_counts().max(),value=init_threshold,
        step=None,marks={str(value) : str(value) for value in df.residue.value_counts().unique()[::-1]}
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
    html.Div(children=[
    html.Footer("2018 - Rob Firth; STFC Hartree Centre; West-Life")
    ], className="row", id="footer-row"),
], className="container")


@app.callback(
    dash.dependencies.Output(component_id="ext_id-output-div", component_property='children'),
    [dash.dependencies.Input(component_id="ext_id-submit-button", component_property="n_clicks")],
    [dash.dependencies.State(component_id='ext_id-input', component_property='value')]
)
def update_output_div(n_clicks, input_ext_id):
    meta = pyre.get_metadata(ext_id=input_ext_id)
    print(meta)
    return ['You\'ve entered {}\n'.format(input_ext_id),
            html.A(html.H4(meta["title"] + " - " + meta["authors"][0]["surname"] + " et al. "+meta["dates"]["accepted"]["year"]),
                   href="https://europepmc.org/articles/"+input_ext_id,
                   target="_blank"),
            html.P(", ".join([i["given_name"][0]+". "+i["surname"] for i in meta["authors"]])),
            dcc.Markdown("__Abstract__"),
            html.P(meta["abstract"]),
            ]

@app.callback(dash.dependencies.Output('intermediate-value', 'children'),
    [dash.dependencies.Input(component_id="ext_id-submit-button", component_property="n_clicks")],
    [dash.dependencies.State(component_id='ext_id-input', component_property='value')]
)
def collect_data(n_clicks, value):
    print(n_clicks, value)
    ext_id = value

    text_dict = pyre.get_text(ext_id)
    fulltext = pyre.reconstruct_fulltext(text_dict, tokenise=False)

    source = pyre.SourceClass()
    source.ext_id = ext_id
    source.text_dict = text_dict
    source.sections = list(zip([text_dict[i]["title"] for i in text_dict], [text_dict[i]["offset"] for i in text_dict]))
    source.fulltext = fulltext

    source.doc = nlp(source.fulltext)

    r = requests.get("https://github.com/RobFirth/pyresid-heroku-dash-app/raw/master/" + ext_id + ".pkl")

    if n_clicks and r.status_code != 200:

        matches = pyre.identify_residues(source.fulltext)

        full_matches = pyre.locate_residues(source, matches, nlp=nlp, decompose=True, verbose=False)

        for match in full_matches:
            match.token = match.token.string
            match.sent = match.sent.string
    else:
        full_matches = pickle.load(io.BytesIO(r.content))

    df = pd.DataFrame([match.__dict__ for match in full_matches])

    return df.to_json(date_format='iso', orient='split')

@app.callback(
    dash.dependencies.Output(component_id='locplot-with-slider', component_property='figure'),
    [dash.dependencies.Input(component_id='intermediate-value', component_property='children'),
     dash.dependencies.Input(component_id='residue-slider', component_property='value')]
)
def update_figure(stored_data, selected_valuecount):
    df=pd.read_json(stored_data, orient="split")

    filtered_df = df[df["residue"].isin(df.residue.value_counts()[df.residue.value_counts() >= selected_valuecount].index.tolist())]

    shapes = []
    annotations = []

    height = len(filtered_df.residue.value_counts()) + 1

    for i, sec in enumerate(source.section_matches):


        shapes.append(
            {"type" : "line",
             'yref': 'paper',
             "x0" : sec.start_token_number,
             "x1" : sec.start_token_number,
             "y0" : 0,
             "y1" : 1,
             "line" : {"width" : 2}}
        )

        if i%2:
            fillcolour = "rgba(255, 255, 255, 0.1)"
        else:
            fillcolour = "rgba(0, 0, 0, 0.1)"

        shapes.append(
            {"type" : "rect",
             "yref" : "paper",
             "x0": sec.start_token_number,
             "x1": sec.end_token_number,
             "y0": 0,
             "y1": 1,
             'fillcolor': fillcolour,
             "line" : {"color" : fillcolour},
             "layer": 'below'
             }
        )

        annotations.append(
            {
            "yref" : "paper",
            "x" : sec.start_token_number,
            "y" : 0.05,
            "text" : sec.EBI_title,
            "showarrow" : False,
            "textangle" : -90,
            "xshift" : 10,
            "font" : {"size" : 16},
            }
        )

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
            xaxis=dict(title="Token Number",
                       showspikes=True),
            yaxis=dict(title='Residue',
                       tickvals=list(np.arange(len(filtered_df.residue.value_counts().index))+1)[::-1],
                       ticktext=[item[0]+" ("+str(item[1])+")" for item in zip([i for i in filtered_df.residue.value_counts().index], [j for j in filtered_df.residue.value_counts()])],
                       showspikes=True),
            margin={'l': "100", 'b': "5%", 't': "0", 'r': "5%", "pad":5},
            hovermode='closest',
            shapes=shapes,
            annotations=annotations
        )
    }


@app.callback(
    dash.dependencies.Output(component_id="residue-slider-div", component_property="children"),
    [dash.dependencies.Input(component_id="intermediate-value", component_property="children")]
)
def update_slider(stored_data):
    df=pd.read_json(stored_data, orient="split")
    return html.Div(dcc.Slider(
        id="residue-slider",
        min = df.residue.value_counts().min(),
        max = df.residue.value_counts().max(),
        value=init_threshold,
        step=None,
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
    return generate_highlight_table(df[df["residue"] == input_value], max_rows=len(df[df["residue"] == input_value]))

# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
