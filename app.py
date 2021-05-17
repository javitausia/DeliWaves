import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64

from datetime import datetime

app = dash.Dash()

server = app.server

data       = pd.read_pickle('data/reconstructed/surfbreaks_reconstructed_final.pkl')
data['Index'] = data['Index'].where(data['Index']<1, 1) * 10
data = data.dropna(how='any', axis=0)

print(data.info())

def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    print(encoded)
    return 'data:image/png;base64,{}'.format(encoded.decode())


message = '''
        En este boceto de app web se van a mostrar y explicar las caracterísiticas principales
        encontradas en las rompientes de la región más conocidas por el surfista local.
        En primer lugar, antes de interaccionar con nada, se muestra un mapa global de las notas
        que se han calculado para las diferentes rompientes, tanto de forma cualitativa, 
        que es la opinión subjetiva de los riders locales, como de forma cuantitativa, a través de 
        un índice de surfeabilidad cuya construcción, así como mayores detalles a cerca 
        de todo lo existente puede consultarse en github/javitausia/DeliWaves o escribiendo a 
        jtausiahoyal@gmail.com o tausiaj@unican.es
        
        
        En estas primeras gráficas, además de este mapa global de notas por rompientes,
        se pueden observar diferentes gráficas con pequeños hexágonos en su interior. Este es el
        resultado conseguido al haber aplicado técnicas de machine learning sobre un conjunto
        de datos cerca del millón de entradas temporales, y no es más que la separación en diferentes grupos de las
        diferentes condiciones de surf que han llegado a la costa de Cantabria durante los
        últimos 40 años
        
        
        Una explicación rápida de los resultados obtenidos en estas gráficas es la siguiente:
        En la figura con colores y flechas, el color del hexágono grande es la altura de ola de rotura,
        el color del hexágono pequeño se asocia con el tipo de rotura, siendo de negro a blanco, de ola
        decrestada (spilling) a cerrote (collapsing), y las flechas indican la dirección del viento (negro, 
        intensidad del mismo relacionada con la longitud de la flecha), y la dirección con la que llegan las olas a la
        playa (color, que dependiendo de este la dispersión o limpieza del oleaje varía)
        
        De este modo, las dos gráficas que siguen al mapa de notas, identificarán las
        condicones de surf y las notas asociadas a cada cluster (grupo), mientras que las gráficas
        de debajo, darán una idea de cúal es la probabilidad de encontrar esos clusters, tanto
        a lo largo del año, como en las distintas rompientes. Más abajo, una vez seleccionada la
        rompiente de interés, aparecerán nuevas gráficas que se comentan a continuación
    '''

message_months = '''
        Como ya se ha comentado, las probabilidades de ocurrencia de los distintos clusters se
        muestran en las dos figuras de encima. Nótese que la realidad cuadra con lo esperado,
        pues los meses de invierno presentan como más probables, aquellos clusters donde las olas
        son más grandes, mientras que los meses de verano presentan oleajes más suaves, y
        claramente influenciados en muchos casos por vientos nordestes. En lo que respecta a las
        rompientes, de nuevo se ve relación clara con las condiciones que los surfistas observan 
        diariamente en sus playas preferidas. A continuación, seleccionar por favor la rompiente
        que se quiera estudiar más con detalle, pudiendo en todo caso cambiar la misma en
        cualquier momento, y pudiendo volver al estado inicial recargando la página web. Una vez se
        seleccione la playa de interés, aparecerán nuevas imágenes, de fácil comprensión por parte
        del lector, pues son parecidas a las previamente mostradas, pero estarán centradas en una
        única rompiente. Además, se muestran perfiles tipo que pueden aparecer en las playas
        a lo largo de los distintos años
    '''
    
message_probs_day = '''
        Una vez se ha seleccionado la rompiente de interés, se muestran tres gráficas dónde puede verse
        la probabilidad de encontrar las distintas condiciones de surf, a lo largo de un año, en los
        diferentes meses/estaciones/años, o en cada mes a lo largo de los años seleccionados
        
        * Nótese que las probabilidades no siempre llegan a 1, aunque esto debería ser así, pero esto
        se debe a la forma en la que se construye la función de agrupación, pues no es del todo completa,
        aunque este bug se solucionará pronto
    '''
    
message_consult_vars = '''
        Por último, se muestra una herramienta de consulta de las diferentes variables existentes en el
        conjunto de datos, donde además se puede seleccionar el intervalo de tiempo que se quiera
    '''
    
top_message  = 'Modelo de reducción de escala híbrido, junto que técnicas de clusterización, que es'
top_message += ' capaz de resolver las condiciones de surf en Cantabria'
        

app.layout = html.Div([html.Div([html.Div([html.Img(src=encode_image('images/app/geoocean.png'),
                                                    style={'height': '150px',
                                                           'width': 'auto',
                                                           'margin-bottom': '10px'})],
                                          className='one-third column'),
                                 html.Div([html.Div([html.H1('Análisis de condiciones de surf con DeliWaves',
                                                             style={'margin-bottom': '5px'}),
                                                     html.H2(top_message, style={'margin-top': '5px'})])],
                                          className='one-half column', id='title'),
                                 html.Div([html.A(html.Button('  github/javitausia/DeliWaves  ', id='learn-more-button'),
                                                  href='https://github.com/javitausia/DeliWaves'),
                                           html.A(html.Button('  Paper oficial (pdf)  ', id='learn-more-pdf'),
                                                  href='https://github.com/javitausia/DeliWaves/blob/master/TausiaHoyalJavier-Surfing.pdf'),
                                           html.A(html.Button('  Resumen ppt (Google Docs)  ', id='learn-more-ppt'),
                                                  href='https://docs.google.com/presentation/d/16N6wJ6zxbf6WDiyEYXN92jBhmhsJwfLXozHU3S2t4LY/edit?usp=sharing')],
                                          id='button', style={'display': 'inline-block'})], 
                                id='header', className='row flex-display', style={'margin-bottom': '25px',
                                                                                  'display':'inline-block'}),
                       html.Hr(),
                       dcc.Markdown(message, style={'fontSize':24}),
                       html.Hr(),
                       html.Div([html.Img(id='map',
                                          src=encode_image('images/app/map-initial.png'),
                                          height=390), 
                                 html.Img(id='image-som', 
                                          src=encode_image('images/app/lasom.png'), 
                                          height=420)]),
                       html.Div([html.Img(id='image-months', 
                                          src=encode_image('images/app/sommonths.png'), 
                                          height=450),
                                 html.Img(id='image-index',
                                          src=encode_image('images/app/sombeaches.png'),
                                          height=440),
                                 html.Hr(),
                                 dcc.Markdown(message_months, style={'fontSize':24})]),
                       html.Hr(),
                       html.Div([html.Div([html.Div(dcc.Dropdown(id='my-variable-both',
                                                                 options=[dict(label=var, value=var) for var in data.columns],
                                                                 multi=False,
                                                                 value='Hs'), 
                                                    style={'width': '30%'}),
                                           html.Div(html.Button(id='submit-button-both',
                                                                n_clicks=0,
                                                                children='Mostrar',
                                                                style={'fontSize':24, 'marginLeft':'30px'}),
                                                    style={'width': '10%'}),
                                           html.Div(html.H3('Pulsa Mostrar para ver resultados en BAR y BOX'),
                                                    style={'width': '40%'})],
                                          style={'display': 'block',
                                                 'align': 'center'}),
                                 html.Div([html.H3('BAR, análisis con diagrama de barras'),
                                           dcc.Dropdown(id='drop-bar',
                                                        options=[{'label': 'Momento del día', 'value': 'Day_Moment'},
                                                                 {'label': 'Hora del día', 'value': 'Hour'},
                                                                 {'label': 'Mes', 'value': 'Month'},
                                                                 {'label': 'Estación', 'value': 'Season'},
                                                                 {'label': 'Año', 'value': 'Year'}],
                                                        value='Month',
                                                        style={'margin-top': '20px'}),       
                                           dcc.Graph(id='graph-bar')], style={'width': '50%', 'display': 'inline-block'}),
                                 html.Div([html.H3('BOX, análisis con diagrama de cajas'),
                                           dcc.Dropdown(id='drop-box',
                                                        options=[{'label': 'TODO', 'value': 'all'},
                                                                 {'label': 'Verano', 'value': 'Summer'},
                                                                 {'label': 'Otoño', 'value': 'Autumn'},
                                                                 {'label': 'Invierno', 'value': 'Winter'},
                                                                 {'label': 'Primavera', 'value': 'Spring'}],
                                                        value='Summer',
                                                        style={'margin-top': '20px'}),       
                                           dcc.Graph(id='graph-box')], style={'width': '50%', 
                                                                              'display': 'inline-block',
                                                                              'align': 'right'})], 
                                style={'width': '98%', 'display': 'inline-block'}),
                       html.Hr(),
                       html.H2('SELECCIONAR la rompiente deseada para analizar sus resultados INDIVIDUALES',
                               style={'width': '50%', 'border': '6px solid #333'}),
                       html.Hr(),
                       html.Div([dcc.RadioItems(
                                    id='radio-sf',
                                    options=[dict(label=sf.upper(), value=sf) for sf in data['beach'].unique()],
                                    value='All',
                                    labelStyle={'display': 'inline-block'})],
                                style={'fontSize':24}),
                       html.Hr(),
                       dcc.Markdown(message_probs_day, style={'fontSize':24}),
                       html.Hr(),
                       dcc.Graph(id='prob-day'),
                       dcc.Dropdown(id='drop-period',
                                    options=[{'label': 'Month', 'value': 'Month'},
                                             {'label': 'Season', 'value': 'Season'},
                                             {'label': 'Year', 'value': 'Year'}],
                                    value='Month',
                                    style={'margin-top': '40px'}),
                       dcc.Graph(id='prob-period'),
                       dcc.RangeSlider(id='range-years',
                                       marks={i: '{}'.format(i) for i in range(data.Year.unique().min(),
                                                                               data.Year.unique().max(),
                                                                               2)},
                                       min=data.Year.unique().min(),
                                       max=data.Year.unique().max(),
                                       value=[2000, 2010]),
                       dcc.Graph(id='prob-time', style={'margin-top': '20px'}),
                       html.Hr(),
                       dcc.Markdown(message_consult_vars, style={'fontSize':24}),
                       html.Hr(),
                       html.Div([html.H3('Selecciona variables y fechas que se quieran:'),
                                 dcc.Dropdown(id='my-variable',
                                              options=[dict(label=var, value=var) for var in data.columns],
                                              multi=True,
                                              value='Hs'), 
                                 dcc.DatePickerRange(id='my-date-picker',
                                                     min_date_allowed=datetime.strptime(
                                                         str(data.index.min())[:10], '%Y-%m-%d'),
                                                     max_date_allowed=datetime.strptime(
                                                         str(data.index.max())[:10], '%Y-%m-%d'),
                                                     start_date=datetime(2010, 1, 1),
                                                     end_date=datetime(2020, 1, 1))], 
                                style={'display':'inline-block'}),
                       html.Div([html.Button(id='submit-button',
                                             n_clicks=0,
                                             children='Mostrar',
                                             style={'fontSize':24, 'marginLeft':'30px'})], 
                                style={'display':'inline-block'}),
                       dcc.Graph(id='time-series')
                      ], 
                      style={'width': '98%'})

@app.callback(Output('map', 'src'),
              [Input('radio-sf', 'value')])
def callback_image_map(radio_sel_sf):
    path = 'images/app/'
    return encode_image(path+radio_sel_sf+'slope.png')

@app.callback(Output('image-som', 'src'),
              [Input('radio-sf', 'value')])
def callback_image_som(radio_sel_sf):
    path = 'images/app/'
    return encode_image(path+'som'+radio_sel_sf+'.png')

@app.callback(Output('image-months', 'src'),
              [Input('radio-sf', 'value')])
def callback_image_months(radio_sel_sf):
    path = 'images/app/'
    return encode_image(path+'som'+radio_sel_sf+'months.png')

@app.callback(Output('image-index', 'src'),
              [Input('radio-sf', 'value')])
def callback_image_index(radio_sel_sf):
    path = 'images/app/'
    return encode_image(path+'som'+radio_sel_sf+'index.png')

@app.callback(Output('graph-bar', 'figure'),
              [Input('submit-button-both', 'n_clicks')],
              [State('my-variable-both', 'value'),
               State('drop-bar', 'value')])
def update_graph_barplot(n_clicks, variable, time):
    return px.bar(data.groupby([time, 'beach']).mean().reset_index(),
                  y=variable,
                  x=time,
                  color='beach',
                  barmode='group',
                  title='Estudio de '+str(variable)+' por '+str(time), 
                  width=900, height=500)

@app.callback(Output('graph-box', 'figure'),
              [Input('submit-button-both', 'n_clicks')],
              [State('my-variable-both', 'value'),
               State('drop-box', 'value')])
def update_graph_boxplot(n_clicks, variable, time):
    data_box = data.copy()
    if isinstance(time, np.int64):
        fig = px.box(data_box.where(data_box['Month']==time).dropna(how='all', axis=0), 
                     x='beach', y=variable,
                     title='Estudio de '+str(variable)+' en '+str(time), 
                     width=1000, height=500)
    elif time=='all':
        fig = px.box(data_box, x='beach', y=variable, title='Points: '+time)
    elif time in ['Winter', 'Spring', 'Summer', 'Autumn']:
        fig = px.box(data_box.where(data_box['Season']==time).dropna(how='all', axis=0), 
                     x='beach', y=variable,
                     title='Estudio de '+str(variable)+' en '+str(time), 
                     width=1000, height=500)
    else:
        fig = px.box(data_box.where(data_box['Day_Moment']==time).dropna(how='all', axis=0), 
                     x='beach', y=variable,
                     title='Estudio de '+str(variable)+' en '+str(time), 
                     width=1000, height=500)
    return fig

@app.callback(Output('prob-day', 'figure'),
              [Input('radio-sf', 'value')])
def callback_fig_prob_day(beach):
    histcolor = ['blue', 'green', 'yellow', 'orange', 'red', 'purple', 'black']
    data_prob = data.where(data['beach']==beach).dropna(how='all', axis=0).copy()
    data_prob = data_prob.groupby([data_prob.index.dayofyear, 
                                   pd.cut(data_prob['Index'],
                                          [0,1,3,5,7,8,9,10],
                                          right=True)])\
                .count().mean(axis=1) / (len(data_prob)/365)
    data_prob.name = 'Probability of RSI'
    
    return px.histogram(data_prob.reset_index(),
                        x='level_0', y='Probability of RSI',
                        color='Index',
                        color_discrete_map={key: value for (key, value) in zip(
                            data_prob.reset_index()['Index'].unique(),
                            histcolor)},
                        nbins=366, range_y=[0,1],
                        labels={'level_0': 'Day of year'},
                        title='Beach: ' + beach, width=1500, height=500)

@app.callback(Output('prob-period', 'figure'),
              [Input('radio-sf', 'value'),
               Input('drop-period', 'value')])
def callback_fig_prob_period(beach, grouper):
    histcolor = ['blue', 'green', 'yellow', 'orange', 'red', 'purple', 'black']
    data_prob = data.where(data['beach']==beach).dropna(how='all', axis=0).copy()
    data_prob = data_prob.groupby([data_prob[grouper], 
                                   pd.cut(data_prob['Index'],
                                          [0,1,3,5,7,8,9,10],
                                          right=True)])\
                .count().mean(axis=1) / (len(data_prob)/len(data[grouper].unique()))
    data_prob.name = 'Probability of RSI'
    return px.histogram(data_prob.reset_index(),
                        x=grouper, y='Probability of RSI',
                        color='Index',
                        color_discrete_map={key: value for (key, value) in zip(
                            data_prob.reset_index()['Index'].unique(),
                            histcolor)},
                        range_y=[0,1], nbins=len(data[grouper].unique()),
                        labels={'level_0': grouper},
                        title='Beach: ' + beach, width=1500, height=500)
    
@app.callback(Output('prob-time', 'figure'),
              [Input('radio-sf', 'value'),
               Input('range-years', 'value')])
def callback_fig_prob_years(beach, years):
    ini_year = years[0]
    end_year = years[1]
    histcolor = ['blue', 'green', 'yellow', 'orange', 'red', 'purple', 'black']
    data_prob = data.where(data['beach']==beach).dropna(how='all', axis=0).copy()
    data_prob = data.where((data['Year']>=ini_year) & (data['Year']<=end_year))\
                .dropna(how='all', axis=0).copy()
    data_prob = data_prob.groupby([pd.Grouper(freq='M'), 
                                   pd.cut(data_prob['Index'],
                                          [0,1,3,5,7,8,9,10],
                                          right=True)])\
                .count().mean(axis=1) / (len(data_prob)/(12*(end_year-ini_year + 1)))
    data_prob.name = 'Probability of RSI'
    return px.histogram(data_prob.reset_index(),
                        x='level_0', y='Probability of RSI',
                        color='Index',
                        color_discrete_map={key: value for (key, value) in zip(
                            data_prob.reset_index()['Index'].unique(),
                            histcolor)},
                        nbins=12*int(end_year-ini_year + 1), range_y=[0,1],
                        labels={'level_0': 'Historical month'},
                        title='Beach: ' + beach, width=1500, height=500)

@app.callback(Output('time-series', 'figure'),
              [Input('submit-button', 'n_clicks')],
              [State('my-variable', 'value'),
               State('radio-sf', 'value'),
               State('my-date-picker', 'start_date'),
               State('my-date-picker', 'end_date')])
def update_graph(n_clicks, variables, beach, start_date, end_date):
    data_series = data.where(data['beach']==beach).dropna(how='all', axis=0).copy()
    return px.line(data_series, x=data_series.index, y=variables, range_x=[start_date[:10],
                                                                           end_date[:10]],
                   title='Beach: ' + beach, width=1500, height=500)

if __name__ == '__main__':
    app.run_server()
