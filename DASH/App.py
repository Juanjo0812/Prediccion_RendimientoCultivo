import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model # type: ignore

# --- Inicializar la app con tema LUX y CSS personalizado ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, '/assets/estilos.css'])
app.title = "Predicción de rendimiento agrícola"

# --- Cargar modelo y preprocesador ---
modelo = load_model("modelo_rendimiento_red.h5")
preprocesador = joblib.load("preprocesador_red.pkl")
le_periodo = joblib.load("label_encoder_periodo.pkl") 
df = pd.read_csv("df_completo.csv")

def codificar_periodo(periodo):
    año = int(periodo[:4])
    sem = 0 if periodo[-1].lower() == 'a' else 1
    return (año - 2006) * 2 + sem  # 2006a → 0, 2006b → 1, etc.

# Coordenadas de municipios
coordenadas = {
    "Algeciras": (2.51, -75.3),
    "La plata": (2.39, -75.86),
    "Gigante": (2.38, -75.53),
    "Garzon": (2.16, -75.63),
    "Neiva": (2.91, -75.15)
}

# --- Listas únicas para el dropdown ---
cultivos = sorted(df["Cultivo_final"].unique())
municipios = sorted(df["Municipio"].unique())

# --- Layout mejorado ---
app.layout = dbc.Container([
    html.H2([
        html.Img(src="/assets/logo.png"),
        "Predicción de Rendimiento Agrícola"
    ], className="text-center mt-4 mb-4"),

    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Label("Municipio"),
                dcc.Dropdown(municipios, id='municipio'),

                dbc.Label("Periodo (ej. 2026a, 2025b)"),
                dcc.Input(id='periodo', type='text', className='form-control'),

                dbc.Label("Área sembrada (ha)"),
                dcc.Input(id='area_sembrada', type='number', className='form-control'),

                dbc.Label("Cultivo"),
                dcc.Dropdown(cultivos, id='cultivo'),

                dbc.Button("Predecir rendimiento", id='btn', color='primary', className='mt-3'),

                html.Div(id='resultado', className='mt-4 text-primary fw-bold')
            ], className="input-panel")
        ], width=4),


        dbc.Col([
            dcc.Graph(id='grafico_rendimiento', style={'height': '350px'}),
            html.Br(),
            dcc.Graph(id='grafico_histograma', style={'height': '350px'}),
            html.Br(),
            dcc.Graph(id='grafico_mapa', style={'height': '350px'})
        ], width=8)
    ])
], fluid=True, className="dash-container")

# --- Callback de predicción y gráficos ---
@app.callback(
    [Output('resultado', 'children'),
     Output('grafico_rendimiento', 'figure'),
     Output('grafico_histograma', 'figure'),
     Output('grafico_mapa', 'figure')],
    Input('btn', 'n_clicks'),
    State('municipio', 'value'),
    State('periodo', 'value'),
    State('area_sembrada', 'value'),
    State('cultivo', 'value')
)


def predecir(n, municipio, periodo, area_sembrada, cultivo):
    if None in [municipio, periodo, area_sembrada, cultivo]:
        return "Por favor complete todos los campos.", go.Figure(), go.Figure(), go.Figure()

    # --- Obtener código DANE y clima ---
    codigo_dane = df[df["Municipio"] == municipio]["Código Dane municipio"].values[0]
    fila = df[(df["Municipio"] == municipio) & (df["Año_periodo"] == periodo) & (df["Cultivo_final"] == cultivo)]

    if not fila.empty:
        clima = fila.iloc[0]
    else:
        clima_base = df[df["Municipio"] == municipio].iloc[0]
        clima = clima_base.copy()
        for col in ['Radiacion_solar', 'Humedad_suelo', 'Precipitacion', 'Humedad_especifica',
                    'Humedad_relativa', 'Temperatura', 'Temperatura_rocio', 
                    'Temperatura_maxima', 'Temperatura_minima', 'Velocidad_viento']:
            clima[col] = max(clima[col] + np.random.normal(0, 0.3), 0.01)

    # --- Crear fila de entrada ---
    nueva_fila = pd.DataFrame([{
        'Código Dane municipio': codigo_dane,
        'Año_periodo_encoded': codificar_periodo(periodo),
        'Área sembrada': area_sembrada,
        'Cultivo_final': cultivo
    }])

    # --- Preprocesamiento ---
    X_nuevo = preprocesador.transform(nueva_fila)

    # --- Predicción ---
    _, pred_rendimiento = modelo.predict(X_nuevo)
    pred = float(pred_rendimiento[0][0])

    promedio_cultivo = df[df["Cultivo_final"] == cultivo]["Rendimiento"].mean()

    # --- Gráfico comparación ---
    fig_barras = go.Figure()
    fig_barras.add_trace(go.Bar(x=["Predicción"], y=[pred], name="Predicción", marker_color='green'))
    fig_barras.add_trace(go.Bar(x=["Promedio"], y=[promedio_cultivo], name="Promedio Histórico", marker_color='gray'))
    fig_barras.update_layout(title="Comparación con promedio histórico del cultivo", yaxis_title="Toneladas por hectárea")

    # --- Gráfico línea ---
    df_linea = df[(df["Municipio"] == municipio) & (df["Cultivo_final"] == cultivo)]
    df_linea_agrupado = df_linea.groupby("Año_periodo")["Rendimiento"].mean().reset_index()

    # Agregar el punto predicho si no existe
    if periodo not in df_linea_agrupado["Año_periodo"].values:
        df_linea_agrupado = pd.concat([
            df_linea_agrupado,
            pd.DataFrame({"Año_periodo": [periodo], "Rendimiento": [pred]})
        ], ignore_index=True)

    # Ordenar correctamente
    df_linea_agrupado = df_linea_agrupado.sort_values("Año_periodo")

    # Crear gráfica
    fig_linea = px.line(df_linea_agrupado, x="Año_periodo", y="Rendimiento", markers=True,
                    title="Evolución del rendimiento con predicción incluida",
                    labels={"Rendimiento": "t/ha", "Año_periodo": "Periodo"})

    # Resaltar el punto predicho
    fig_linea.add_scatter(x=[periodo], y=[pred], mode='markers+text', 
                        marker=dict(size=12, color='green'),
                        text=["Predicción"],
                        textposition="top center",
                        name="Predicción futura")

    # --- Mapa ---
    lat, lon = coordenadas.get(municipio, (2.5, -75.5))
    fig_map = px.scatter_mapbox(lat=[lat], lon=[lon], zoom=8,
                                 mapbox_style="carto-positron", height=300,
                                 title=f"Ubicación del municipio: {municipio}")

    texto = f"\U0001F33E El rendimiento estimado es de {pred:.2f} t/ha"
    return texto, fig_barras, fig_linea, fig_map

if __name__ == "__main__":
    app.run(debug=True)
