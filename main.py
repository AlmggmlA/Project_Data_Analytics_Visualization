import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import pydeck as pdk

st.title("Projeto Final - Análise Gráfica")

df_salary_IT = pd.read_csv('./data/data_science_salary_21_cols.csv',index_col = 0)

df_columns = df_salary_IT.columns
col1,col2 = st.columns(2)

with col1:
    eixo_X = st.selectbox("eixo X:", df_columns)
    eixo_Y = st.selectbox("eixo Y:", df_columns)
    if (eixo_X != eixo_Y):
        col1.dataframe(df_salary_IT[[eixo_X,eixo_Y]].head())
    else:
        col1.dataframe(df_salary_IT.head())

# PIE - matplotlib
def grafico_genero_porcentagem():
    df_qtd = df_salary_IT.groupby(df_salary_IT['Genero']).size()
    df_qtd_genero = df_qtd.to_frame()
    df_qtd_genero.rename(columns={0: 'qtd'}, inplace=True)
    grafico, eixo = plt.subplots()
    eixo.pie(x='qtd',
             autopct="%.2f%%",
             explode=[0.08, 0.05],
             labels=['Mulheres', 'Homens'],
             data=df_qtd_genero)
    col2.pyplot(grafico)

# BAR - plotly
def grafico_genero_mudouEstado():
    df_filtro = df_salary_IT.groupby(['Genero', "Mudou de Estado?"], as_index=False).size()

    df_gen_mudou_estado_ctb = pd.crosstab(index = df_filtro['Genero'],
                                          columns = df_filtro['Mudou de Estado?'],
                                          values = df_filtro['size'],
                                          aggfunc = "sum")

    # gera gráfico, separado por colunas
    grafico = px.bar(data_frame = df_gen_mudou_estado_ctb,
                     x = df_gen_mudou_estado_ctb.index,
                     y = df_gen_mudou_estado_ctb.columns,
                     base = df_gen_mudou_estado_ctb.index,
                     barmode="group")

    #grafico.update_traces(name = "não") # coloca um nome dentro do quadro de legenda
    col2.plotly_chart(grafico)

# BAR - plotly
def grafico_genero_salario():
    df_filtro = df_salary_IT.groupby(['Genero', 'Faixa salarial'], as_index=False).size()

    df_genero_salario_ctb = pd.crosstab(index = df_filtro['Genero'],
                                        columns = df_filtro['Faixa salarial'],
                                        values = df_filtro['size'],
                                        aggfunc = 'sum')

    grafico = px.bar(data_frame=df_genero_salario_ctb,
                     x=df_genero_salario_ctb.index,
                     y=df_genero_salario_ctb.columns,
                     base=df_genero_salario_ctb.index,
                     barmode="group")
    col2.plotly_chart(grafico)

def grafico_uf_genero():
    grafico = sns.displot(x='uf onde mora', col="Genero", data=df_salary_IT)
    col2.pyplot(grafico)

def grafico_regiao_genero():
    grafico = sns.displot(x='Regiao onde mora', col="Genero", data=df_salary_IT)
    col2.pyplot(grafico)


def show_column_map(data):
    data['lat'] = data['uf onde mora'].apply(_get_coord, args=('lat',))
    data['lon'] = data['uf onde mora'].apply(_get_coord, args=('lon',))

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=-23.901,
            longitude=-46.525,
            zoom=5,
            pitch=50
        ),
        layers=[
            pdk.Layer(
                'ColumnLayer',
                data=data,
                get_position='[lon, lat]',
                #get_elevation='[newCases+newDeaths]',
                radius=20000,
                auto_highlight=True,
                elevation_scale=100,
                elevation_range=[0, 5000],
                pickable=True,
                extruded=True,
                #get_color="[10 10 10 255]"
            ),
            pdk.Layer(
                'ColumnLayer',
                data=data,
                get_position='[lon, lat]',
                #get_elevation='[newCases]',
                radius=20000,
                auto_highlight=True,
                elevation_scale=100,
                elevation_range=[0, 5000],
                pickable=True,
                extruded=True,
                #get_color="[128 10 10 255]"
            )
        ]
    ))

def _get_coord(state, orientation):
    try:
        return STATES_COORD[state][orientation]
    except:
        return 0

STATES_COORD = {
    'AC': {'lat': -9.59, 'lon': -70.09},
    'AL': {'lat': -9.63, 'lon': -36.11},
    'AM': {'lat': -4.52, 'lon': -62.76},
    'AP': {'lat': 0.94, 'lon': -51.33},
    'BA': {'lat': -12.93, 'lon': -40.97},
    'CE': {'lat': -5.27, 'lon': -39.18},
    'DF': {'lat': -15.86, 'lon': -47.88},
    'ES': {'lat': -19.83, 'lon': -40.29},
    'GO': {'lat': -15.69, 'lon': -49.81},
    'MA': {'lat': -4.71, 'lon': -44.57},
    'MG': {'lat': -19.21, 'lon': -44.18},
    'MS': {'lat': -20.65, 'lon': -54.75},
    'MT': {'lat': -13.07, 'lon': -56.61},
    'PA': {'lat': -5.89, 'lon': -52.42},
    'PB': {'lat': -7.26, 'lon': -36.04},
    'PE': {'lat': -8.75, 'lon': -37.66},
    'PI': {'lat': -6.86, 'lon': -42.94},
    'PR': {'lat': -24.85, 'lon': -51.11},
    'RJ': {'lat': -22.51, 'lon': -42.67},
    'RO': {'lat': -11.56, 'lon': -62.47},
    'RR': {'lat': -1.12, 'lon': -61.25},
    'RS': {'lat': -29.64, 'lon': -52.89},
    'SC': {'lat': -27.33, 'lon': -50.02},
    'SE': {'lat': -10.71, 'lon': -37.35},
    'SP': {'lat': -22.49, 'lon': -48.15},
    'TO': {'lat': -10.21, 'lon': -47.91}
}
def grafico_mapa_uf():
    df_estado = pd.DataFrame(df_salary_IT['uf onde mora'])
    df_estado.dropna(axis=0, how="any", inplace=True)

    show_column_map(df_estado)

with col2:

    if(eixo_X == 'Genero'):
        if (eixo_Y == 'Mudou de Estado?'):
            grafico_genero_mudouEstado()
        elif (eixo_Y == 'Faixa salarial'):
            grafico_genero_salario()
        else:
            grafico_genero_porcentagem()
    elif (eixo_Y == 'Genero'):
        if (eixo_X == 'uf onde mora'):
            grafico_uf_genero()
        elif (eixo_X == 'Regiao onde mora'):
            grafico_regiao_genero()
    elif(eixo_X == 'uf onde mora'):
        grafico_mapa_uf()




