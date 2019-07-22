# Import Supporting Libraries
import pandas as pd

# Import Dash Visualization Libraries
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash.dependencies
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

# Load datasets
train = pd.read_csv('https://raw.githubusercontent.com/tiagoportelanelo/nelotest/master/train.csv', index_col ='PassengerId' )

df_describe = train.describe().copy()
df_describe.insert(0, 'Stat', train.describe().index.to_list())

df_aux = train.copy()
df_aux['Sex'] = df_aux.Sex.map({'female':0, 'male':1})

df_corr = df_aux.corr().copy()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([

    html.H1('Title'),
    html.H2('DataFrame Overview'),
    html.H4('Primeiras entradas do DataSet'),
    dt.DataTable(
        id='table_head',
        columns=[{"name": i, "id": i} for i in train.head().columns],
        data=train.head().to_dict('records'),
    ),
    
    html.H4('Descricao Estatistica'),

    dt.DataTable(
        id='table_describe',
        columns=[{"name": i, "id": i} for i in df_describe.columns],
        data=df_describe.to_dict('records'),
    ),

    html.H4('Matriz de Correlacao'),
    dcc.Graph(
        id='crr-matrix',
        
        figure = go.Figure(data=go.Heatmap(
                    z=[df_corr.Survived,
                      df_corr.Pclass,
                      df_corr.Sex,
                      df_corr.Age, 
                      df_corr.SibSp,
                      df_corr.Parch,
                      df_corr.Fare,
                      ],
                    x =['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'],
                    y =['Survived', 'Pclass' ,'Sex','Age', 'SibSp', 'Parch', 'Fare']
                    ))
        
    ),

    html.H2('Analise Dos Atributos'),


    html.Div(id='selected-indexes'),
    dcc.Dropdown(
        id='atributes-list',
        options=[
            {'label': i, 'value': i} for i in train.columns
        ],
        value = 'Survived'
    ),
    dcc.Graph(id='var-plot'),

], style={'width': '60%'})


@app.callback(Output('var-plot', 'figure'),
              [Input('atributes-list', 'value')])
def update_figure(value):
        if value == 'Survived':
            train.Survived.value_counts(normalize=True)

            figure={
                'data': [
                    {'x': list(train.Survived.value_counts(normalize=True).index), 'y': train.Survived.value_counts(normalize=True).to_list(), 'type': 'bar', 'name': 'SF'}
                ],
            }
    
            return figure

        if value == 'Sex':
            train.Survived.value_counts(normalize=True)

            figure={
                'data': [
                    {'x': list(train.Sex.value_counts(normalize=True).index), 'y': train.Sex.value_counts(normalize=True).to_list(), 'type': 'bar', 'name': 'SF'}
                ],
            }
    
            return figure
        
        if value == 'Pclass':
            train.Survived.value_counts(normalize=True)

            figure={
                'data': [
                    {'x': list(train.Pclass.value_counts(normalize=True).index), 'y': train.Pclass.value_counts(normalize=True).to_list(), 'type': 'bar', 'name': 'SF'}
                ],
            }
    
            return figure
        if value == 'Age':
            trace = go.Histogram(x=train["Age"], opacity=0.7, name="Male",
                         xbins={"size": 5}, customdata=train["Age"], )
            layout = go.Layout(title="Age Distribution", xaxis={"title": "Age (years)", "showgrid": False},
                       yaxis={"title": "Count", "showgrid": False}, )
            figure = {"data": [trace], "layout": layout}
    
            return figure

if __name__ == '__main__':
    app.run_server(debug=True)