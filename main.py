import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import pydeck as pdk

st.markdown('<h1 style="text-align: center;">Projeto Final - Análise Gráfica</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">- Profissionais de TI no Brasil em 2021 -</h3>', unsafe_allow_html=True)
st.markdown('<br/><br/><br/>', unsafe_allow_html=True)


colunas = ['Genero','Mudou de Estado?','Faixa salarial','uf onde mora','Regiao onde mora','Nivel de Ensino','Faixa idade','Idade']
df_salary_IT = pd.read_csv('./data/data_science_salary_21_cols.csv', usecols=colunas)

df_columns = df_salary_IT.columns
col1,col2 = st.columns(2)

with col1:
    eixo_X = st.selectbox("eixo X:", df_columns)
    eixo_Y = st.selectbox("eixo Y:", df_columns)
    # if (eixo_X != eixo_Y):
    #     pass
        #col1.dataframe(df_salary_IT[[eixo_X,eixo_Y]].head())
    # else:
    #     pass
        #col1.dataframe(df_salary_IT.head())

# PIE - matplotlib
def grafico_genero_porcentagem():
    st.markdown('<h6 style="text-align: center;">Percentual por Gênero</h6>',
                unsafe_allow_html=True)
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
    st.markdown('<h4 style="text-align: center;">Mudaram de Estado</h4>',
                unsafe_allow_html=True)
    df_filtro = df_salary_IT.groupby(['Genero', "Mudou de Estado?"], as_index=False).size()

    df_gen_mudou_estado_ctb = pd.crosstab(index = df_filtro['Genero'],
                                          columns = df_filtro['Mudou de Estado?'],
                                          values = df_filtro['size'],
                                          aggfunc = "sum")
    df_gen_mudou_estado_ctb.rename(columns = {0:"Não", 1:"Sim"},inplace=True)
    # gera gráfico, separado por colunas
    grafico = px.bar(data_frame = df_gen_mudou_estado_ctb,
                     x = df_gen_mudou_estado_ctb.index,
                     y = df_gen_mudou_estado_ctb.columns,
                     base = df_gen_mudou_estado_ctb.index,
                     barmode="group")

    col1.dataframe(df_gen_mudou_estado_ctb)
    col2.plotly_chart(grafico)

# BAR - plotly
def grafico_genero_salario():
    st.markdown('<h4 style="text-align: center;">Divisão por Faixa Salarial e Gênero</h4>',
                unsafe_allow_html=True)
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
    col1.dataframe(df_genero_salario_ctb)
    col2.plotly_chart(grafico)

def grafico_uf_genero():
    st.markdown('<h5 style="text-align: center;">Divisão por Estado e Gênero</h5>',
                unsafe_allow_html=True)
    grafico = sns.displot(x='uf onde mora', col="Genero", data=df_salary_IT)
    col2.pyplot(grafico)

def grafico_regiao_genero():
    st.markdown('<h5 style="text-align: center;">Divisão por Região e Gênero</h5>',
                unsafe_allow_html=True)
    grafico = sns.displot(x='Regiao onde mora', col="Genero", data=df_salary_IT)
    col2.pyplot(grafico)

def show_column_map(data):
    data['lat'] = data['UF'].apply(_get_coord, args=('lat',))
    data['lon'] = data['UF'].apply(_get_coord, args=('lon',))

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
                'HeatmapLayer',
                data=data,
                get_position='[lon, lat]',
                get_elevation='QTD',
                radius=20000,
                auto_highlight=True,
                elevation_scale=100,
                elevation_range=[0, 5000],
                pickable=True,
                extruded=True,
                get_color="[10 10 10 255]"
            ),
            pdk.Layer(
                'ColumnLayer',
                data=data,
                get_position='[lon, lat]',
                get_elevation='QTD',
                radius=20000,
                auto_highlight=True,
                elevation_scale=100,
                elevation_range=[0, 5000],
                pickable=True,
                extruded=True,
                get_color="[180, 0, 200, 140]"
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
    st.markdown('<h6 style="text-align: center;">Quantidade de Profissionais de TI por Estado</h6>',
                unsafe_allow_html=True)
    df_estado = pd.DataFrame(df_salary_IT['uf onde mora'])
    df_estado.dropna(axis=0, how="any", inplace=True)
    df_estado_filtro = df_estado.groupby(['uf onde mora'], as_index=False).size()
    df_estado_filtro.rename(columns={"size":"QTD","uf onde mora":"UF"},inplace=True)
    col1.dataframe(df_estado_filtro)
    show_column_map(df_estado_filtro)

def grafico_salario_genero():
    # criando gráfico de pirâmide para salários
    abs_genero = pd.crosstab(df_salary_IT["Faixa salarial"], df_salary_IT["Genero"])

    women_pop = list(abs_genero.Feminino)
    men_pop = list(abs_genero.Masculino)
    men_pop = [element * -1 for element in men_pop]

    faixa = list(abs_genero.index.values)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=faixa,
        x=men_pop,
        name='Masculino',
        marker_color='lightgray',
        orientation='h'
    ))
    fig.add_trace(go.Bar(
        y=faixa,
        x=women_pop,
        name='Feminino',
        marker_color='#710c04',
        orientation='h'
    ))

    fig['layout']['title'] = "Pirâmide salarial segundo o gênero"
    fig['layout']['yaxis']['title'] = 'Faixa salarial'

    fig.update_layout(barmode='relative', xaxis_tickangle=-90)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=800, height=500)
    #st.subheader('Pirâmide salarial segundo o gênero')
    col2.plotly_chart(fig)

def grafico_escolaridade_genero():
    # Criando subplots: usando 'domain' para Pie subplot
    st.markdown('<h4 style="text-align: center;">Nível de Escolaridade por Gênero</h4>',
                unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])

    # MASCULINO
    labels_male = df_salary_IT["Nivel de Ensino"].where(df_salary_IT["Genero"] == 'Masculino').value_counts().index.values
    values_male = df_salary_IT["Nivel de Ensino"].where(df_salary_IT["Genero"] == 'Masculino').value_counts().values

    # FEMININO
    labels_female = df_salary_IT["Nivel de Ensino"].where(df_salary_IT["Genero"] == 'Feminino').value_counts().index.values
    values_female = df_salary_IT["Nivel de Ensino"].where(df_salary_IT["Genero"] == 'Feminino').value_counts().values

    # gráfico
    # Add traces
    fig.add_trace(
        go.Pie(
            values=values_male,
            labels=labels_male,
            hole=0.6,
            title='Nível de ensino dos Homens'
        ),
        1, 1
    )

    fig.add_trace(
        go.Pie(
            values=values_female,
            labels=labels_female,
            hole=0.6,
            title='Nível de ensino das Mulheres'
        ),
        1, 2
    )

    fig.add_annotation(
        dict(
            font=dict(color='#400080', size=10),
            x=0.56,
            y=-0.05,
            showarrow=False,
            textangle=0,
            xanchor='right',
            xref="paper",
            yref="paper"
        )
    )

    fig.add_annotation(
        dict(
            font=dict(color='#400080', size=10),
            x=1.12,
            y=-0.05,
            showarrow=False,
            textangle=0,
            xanchor='right',
            xref="paper",
            yref="paper"
        )
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    col2.plotly_chart(fig)

def grafico_faixaIdade():
    df_salary_gen_age = df_salary_IT.groupby('Faixa idade')['Genero'].count()
    grafico, eixo = plt.subplots()
    eixo = df_salary_gen_age.plot(kind='bar')
    col2.pyplot(grafico)

def grafico_idade():
    grafico, eixo = plt.subplots()
    eixo = df_salary_IT.boxplot('Idade')
    col2.pyplot(grafico)

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
        elif (eixo_X == 'Faixa salarial'):
            grafico_salario_genero()
        elif (eixo_X == 'Nivel de Ensino'):
            grafico_escolaridade_genero()
    elif(eixo_X == 'uf onde mora'):
        grafico_mapa_uf()
    elif (eixo_X == 'Faixa idade'):
        grafico_faixaIdade()
    elif (eixo_X == 'Idade'):
        grafico_idade()
