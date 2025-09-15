from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

                     
class ComponentGenerator:
    
    def __init__(self):
        pass
    
    def input_dropdown_pair(self):
        return [
            dcc.Input(id='ticker-1', type='text', value='SPY'),
            dcc.Dropdown(['Long', 'Short'], 'Long',id='direction')
        ]

class Application:
    def __init__(self):
        self.external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        self.cg = ComponentGenerator()
        app.layout = self.set_layout()
    
        
    def run_server(self,mode='app'):
        if mode == 'inline':
            app.run_server(mode="inline")
        elif mode== 'app':
            app.run_server()
        
    def set_layout(self):
        return html.Div([
            html.Div(id='container'),
            html.Button(id='button', n_clicks=0, children='Submit'),
        ])

class Application_Portfolio(Application):
    def __init__(self):
        pass
    
            
@app.callback(Output('container',  'children'), Input('button','n_clicks'))
def append_pair(n_clicks):
    my_list = []
    for i in range(n_clicks):
        my_list.append(html.Div(cg.input_dropdown_pair()))
                    
    return my_list

    
 