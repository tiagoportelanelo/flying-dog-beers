import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.graph_objs as go
# Load datasets
train = pd.read_csv('https://github.com/tiagoportelanelo/nelotest/blob/master/test.csv', index_col ='PassengerId' )

df_describe = train.describe().copy()
df_describe.insert(0, 'Stat', train.describe().index.to_list())

df_corr = train.corr().copy()

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
                      df_corr.Age, 
                      df_corr.SibSp,
                      df_corr.Parch,
                      df_corr.Fare],
                    x =['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
                    y =['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
                    ))
        
    ),

    html.H2('Analise Dos Atributos'),


    html.Div(id='selected-indexes'),
    dcc.Dropdown(
        id='atributes-list',
        options=[
            {'label': i, 'value': i} for i in train.columns
        ],
    ),
    dcc.Graph(id='var-plot'),

], style={'width': '60%'})

if __name__ == '__main__':
    app.run_server()
