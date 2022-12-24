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
st.markdown('<h3 style="text-align: center;">- Profissionais de TI no Brasil em 2021* -</h3>', unsafe_allow_html=True)
st.markdown('<br/><br/><br/>', unsafe_allow_html=True)

st.markdown('<h8 style="text-align: center;">**Autores:**  <font color=blue>Aprígio Gusmão e Fausto Lucena</font> </h8>', unsafe_allow_html=True)
st.markdown('<h7 style="text-align: center;">*Fonte:*  <font color=blue>https://www.kaggle.com/datasets/datahackers/state-of-data-2021</font> </h7>', unsafe_allow_html=True)

colunas = ['Genero','Mudou de Estado?','Faixa salarial','uf onde mora','Regiao onde mora','Nivel de Ensino','Faixa idade','Idade']
df_salary_IT = pd.read_csv('./data/data_science_salary_21_cols.csv', usecols=colunas)

df_columns = df_salary_IT.columns
lst_tipo_analise = ['Mudou de Estado x Gênero','Salário por gênero','Porcentagem por gênero',
                    'Gênero por Estado','Gênero por Região', "Gênero por Escolaridade",
                    'Mapeamento por Estado','Faixa Etária','Responderam questionário por Estado']
col1,col2 = st.columns(2)

with col1:
    # eixo_X = st.selectbox("eixo X:", df_columns)
    # eixo_Y = st.selectbox("eixo Y:", df_columns)
    tipo_analise = st.selectbox("Tipo de análise:",lst_tipo_analise)
    st.text('*de acordo com o questionário preenchido \nno período de 18 de outubro de 2021 a \n6 de dezembro de 2021.')

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

    #df_filtro = df_salary_IT.groupby(['Genero', "Mudou de Estado?"], as_index=False).size()
    # valor = []
    # df_size_gen = df_filtro['size']
    # for indice in range(len(df_filtro['Genero'])):
    #     if df_filtro['Genero'][indice] == 'Feminino':
    #         valor.append(df_size_gen[indice] / df_size_gen[0:2].sum())
    #     else:
    #         valor.append(df_size_gen[indice] / df_size_gen[2:4].sum())
    # porcentagem = pd.DataFrame(valor)
    # df_gen_mudou_estado = df_filtro

    # df_gen_mudou_estado['porcentagem'] = porcentagem[0].apply(lambda x : f"{x:.2f}%")
    # df_gen_mudou_estado['porcentagem'] = porcentagem[0].apply(lambda x: x)

    # df_gen_mudou_estado_ctb = pd.crosstab(index = df_gen_mudou_estado['Genero'],
    #                                       columns = df_gen_mudou_estado['Mudou de Estado?'],
    #                                       values = df_gen_mudou_estado['size'],
    #                                       aggfunc = 'sum')
    # df_gen_mudou_estado_ctb.rename({0:"Não", 1:"Sim"}, axis = 1, inplace=True)


    df_filtro = df_salary_IT.groupby(['Genero', 'Mudou de Estado?']).size().reset_index()
    df_filtro['porcentagem'] = df_salary_IT.groupby(['Genero','Mudou de Estado?']
    ).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
    df_filtro['porcentagem'] = df_filtro['porcentagem'].map('{:,.2f}%'.format)

    df_gen_mudou_estado_ctb = df_filtro
    # gera gráfico, separado por colunas
    grafico = px.bar(data_frame=df_gen_mudou_estado_ctb,
                     x=df_gen_mudou_estado_ctb['Genero'],
                     y=df_gen_mudou_estado_ctb['porcentagem'],
                     color='porcentagem',
                     color_discrete_sequence=[px.colors.qualitative.Safe[10],
                                              px.colors.qualitative.Safe[9],
                                              px.colors.qualitative.Safe[10],
                                              px.colors.qualitative.Safe[9],],
                     text = df_gen_mudou_estado_ctb['porcentagem'],
                     base=df_gen_mudou_estado_ctb['Genero'],
                     barmode="group")
    grafico.update(layout_showlegend=False)
    grafico.add_annotation(text='Profissionais de TI',
                           font=dict(
                               family="Courier New",
                               size=18,
                               color="black"
                           ), showarrow=False)


    df_gen_md_estado = pd.crosstab(index= df_filtro['Genero'],
                                   columns=df_filtro['Mudou de Estado?'],
                                   values=df_filtro['porcentagem'],
                                   aggfunc = 'sum')

    df_gen_md_estado.rename(columns={0: "Não", 1: "Sim"}, inplace=True)


    #df_gen_mudou_estado_ctb.rename(columns={0: "Não", 1: "Sim"}, inplace=True)

    # df_gen_mudou_estado_ctb = pd.crosstab(index = df_filtro['Genero'],
    #                                       columns = df_filtro['Mudou de Estado?'],
    #                                       values = df_filtro['size'],
    #                                       aggfunc = "sum")
    #df_gen_mudou_estado_ctb.rename(columns = {0:"Não", 1:"Sim"},inplace=True)

    # gera gráfico, separado por colunas
    # grafico = px.bar(data_frame = df_gen_mudou_estado_ctb,
    #                  x = df_gen_mudou_estado_ctb.index,
    #                  y = df_gen_mudou_estado_ctb.columns,
    #                  base = df_gen_mudou_estado_ctb.index,
    #                  barmode="group")

    col1.dataframe(df_gen_md_estado)
    col2.plotly_chart(grafico)

# BAR - plotly
def grafico_genero_salario():
    st.markdown('<h4 style="text-align: center;">Divisão por Faixa Salarial e Gênero</h4>',
                unsafe_allow_html=True)
    df_indice = ['Menos de R$ 1.000/mês',
                 'de R$ 1.001/mês a R$ 2.000/mês',
                 'de R$ 2.001/mês a R$ 3000/mês',
                 'de R$ 3.001/mês a R$ 4.000/mês',
                 'de R$ 4.001/mês a R$ 6.000/mês',
                 'de R$ 6.001/mês a R$ 8.000/mês',
                 'de R$ 8.001/mês a R$ 12.000/mês',
                 'de R$ 12.001/mês a R$ 16.000/mês',
                 'de R$ 16.001/mês a R$ 20.000/mês',
                 'de R$ 20.001/mês a R$ 25.000/mês',
                 'de R$ 25.001/mês a R$ 30.000/mês',
                 'de R$ 30.001/mês a R$ 40.000/mês',
                 'Acima de R$ 40.001/mês']

    # criando gráfico de pirâmide para salários
    df_faixaSalarial_genero = pd.DataFrame(df_salary_IT, columns=['Faixa salarial', 'Genero'])
    df_faixaSalarial_genero.dropna(axis=0, how='any', inplace=True)
    df_faixaSalarial_genero["posicao"] = [faixa_salarial
                                          for faixa_salarial in df_indice
                                          for faixa in df_faixaSalarial_genero['Faixa salarial']
                                          if faixa_salarial == faixa]
    df_faixaSalarial_genero.sort_values(by='posicao', inplace=True)
    df_genero_salario_ctb = pd.crosstab(df_faixaSalarial_genero["Faixa salarial"],
                                        df_faixaSalarial_genero["Genero"]).reindex(df_indice)

  # Estruturando o gráfico
    women_pop = list(df_genero_salario_ctb.Feminino)
    men_pop = list(df_genero_salario_ctb.Masculino)
    faixa = list(df_genero_salario_ctb.index.values)

    #fig = make_subplots(rows=1, cols=2)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=men_pop,
        x=faixa,
        name='Masculino',
        marker_color='lightgray',
    ))
    fig.add_trace(go.Bar(
        y=women_pop,
        x=faixa,
        name='Feminino',
        marker_color='#710c04',
    ))


    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=800, height=500)
    fig.update_layout(margin_b=180)

    col2.plotly_chart(fig)

    # Calculando porcentangem
    df_porcentagem = pd.DataFrame(df_genero_salario_ctb)
    total_mulheres = df_porcentagem['Feminino'].sum()
    df_porcentagem['Fem (%)'] = df_porcentagem['Feminino'].apply(lambda x: 100 * (x / total_mulheres)).values
    df_porcentagem['Fem (%)'] = df_porcentagem['Fem (%)'].map('{:,.2f}%'.format)

    total_homens = df_porcentagem['Masculino'].sum()
    df_porcentagem['Masc (%)'] = df_porcentagem['Masculino'].apply(lambda x: 100 * (x / total_homens)).values
    df_porcentagem['Masc (%)'] = df_porcentagem['Masc (%)'].map('{:,.2f}%'.format)
    col1.dataframe(df_porcentagem[['Fem (%)', 'Masc (%)']])
    st.write(f'Total de mulheres: {total_mulheres}')
    st.write(f'Total de homens: {total_homens}')


    #st.dataframe(df_faixaSalarial_genero['Faixa salarial'])
    #df_filtro = df_salary_IT.groupby(['Genero', 'Faixa salarial'], as_index=False).size()
    # df_genero_salario_ctb = pd.crosstab(index = df_filtro['Genero'],
    #                                     columns = df_filtro['Faixa salarial'],
    #                                     values = df_filtro['size'],
    #                                     aggfunc = 'sum')
    #
    # grafico = px.bar(data_frame=df_genero_salario_ctb,
    #                  x=df_genero_salario_ctb.index,
    #                  y=df_genero_salario_ctb.columns,
    #                  base=df_genero_salario_ctb.index,
    #                  barmode="group")

    #col1.dataframe(df_genero_salario_ctb)

    #col2.plotly_chart(grafico)

def grafico_uf_genero():
    st.markdown('<h5 style="text-align: center;">Divisão por Estado e Gênero</h5>',
                unsafe_allow_html=True)

    # paleta de cores
    lst_cores = ["#BF45F5","#9A77A8","#AB6748", "#FA601B", "#42F55D", "#66dee2","#A67386",
                 "#733049","#D3C5D6","#D2D943","#A6A279","#090AB3","#313145","#000000",
                 "#07330A","#4E9163","#FF9CFF","#C3D91C","#FF4336","#FFF49C","#00A8FF"]

    grafico = sns.displot(x='uf onde mora',
                          col="Genero",
                          hue="uf onde mora",
                          palette=lst_cores,
                          data=df_salary_IT)
    grafico.set_xticklabels(rotation= -45)
    #sns.set(rc = {'figure.figsize':(25,19)})
    #grafico.figure(figsize=(25,19))
    col2.pyplot(grafico)

def grafico_regiao_genero():
    st.markdown('<h5 style="text-align: center;">Divisão por Região e Gênero</h5>',
                unsafe_allow_html=True)

    # paleta de cores
    lst_cores = ["#BF45F5", "#9A77A8", "#AB6748", "#FA601B", "#42F55D", "#66dee2"]

    # Gráfico
    grafico = sns.displot(x='Regiao onde mora',
                          col = "Genero",
                          hue="Regiao onde mora",
                          palette=lst_cores,
                          data=df_salary_IT)
    grafico.set_xticklabels(rotation= -45)
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
                get_elevation='elevacao',
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
                get_elevation='elevacao',
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
    df_estado_filtro['valor percentual'] = df_estado_filtro['QTD'].apply(lambda x: f"{(x/df_estado_filtro['QTD'].sum())*100:.2f}%")
    df_estado_filtro['elevacao'] = df_estado_filtro['valor percentual'].apply(lambda x:float(x.replace('%',''))*50)
    col1.dataframe(df_estado_filtro[['UF','QTD','valor percentual']])
    show_column_map(df_estado_filtro)

def grafico_salario_genero():

    df_indice = ['Menos de R$ 1.000/mês',
                 'de R$ 1.001/mês a R$ 2.000/mês',
                 'de R$ 2.001/mês a R$ 3000/mês',
                 'de R$ 3.001/mês a R$ 4.000/mês',
                 'de R$ 4.001/mês a R$ 6.000/mês',
                 'de R$ 6.001/mês a R$ 8.000/mês',
                 'de R$ 8.001/mês a R$ 12.000/mês',
                 'de R$ 12.001/mês a R$ 16.000/mês',
                 'de R$ 16.001/mês a R$ 20.000/mês',
                 'de R$ 20.001/mês a R$ 25.000/mês',
                 'de R$ 25.001/mês a R$ 30.000/mês',
                 'de R$ 30.001/mês a R$ 40.000/mês',
                 'Acima de R$ 40.001/mês']

    # criando gráfico de pirâmide para salários
    df_faixaSalarial_genero = pd.DataFrame(df_salary_IT, columns= ['Faixa salarial','Genero'])
    df_faixaSalarial_genero.dropna(axis=0, how='any', inplace=True)
    df_faixaSalarial_genero["posicao"] = [faixa_salarial
                                          for faixa_salarial in df_indice
                                          for faixa in df_faixaSalarial_genero['Faixa salarial']
                                          if faixa_salarial == faixa]
    df_faixaSalarial_genero.sort_values(by='posicao', inplace=True)

    abs_genero = pd.crosstab(df_faixaSalarial_genero["Faixa salarial"],
                             df_faixaSalarial_genero["Genero"]).reindex(df_indice)

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
    fig.update_layout(margin_l=200)
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
    st.markdown('<h4 style="text-align: center;">Faixa Etária</h4>',
                unsafe_allow_html=True)
    df_salary_gen_age = df_salary_IT.groupby('Faixa idade')['Genero'].count()
    grafico, eixo = plt.subplots()
    eixo = df_salary_gen_age.plot(kind='bar')
    col2.pyplot(grafico)

def grafico_idade():
    st.markdown('<h4 style="text-align: center;">Idade</h4>',
                unsafe_allow_html=True)
    grafico, eixo = plt.subplots()
    eixo = df_salary_IT.boxplot('Idade')
    col2.pyplot(grafico)

def grafico_questionario_uf():
    #df101 = df_salary_IT.groupby(['Nivel'])['Genero', 'uf onde mora', 'Faixa salarial'].value_counts()
    qtd = df_salary_IT['Regiao onde mora'].value_counts().values
    idx = df_salary_IT['Regiao onde mora'].value_counts().index

    # paleta de cores
    lst_cores = ["#BF45F5","#9A77A8","#AB6748", "#FA601B", "#42F55D", "#66dee2"]

    plt.figure(figsize=(24, 8))
    grafico01,eixo01 = plt.subplots() #(1, 2, 1)
    eixo01 = plt.bar(idx, qtd, ec="k", alpha=.6, color= lst_cores)
    plt.xlabel('Estado')
    plt.title("Quantidade de profissionais que preencheram o questionario por UF")
    grafico02,eixo02 = plt.subplots() #(1, 2, 2)
    eixo02 = plt.pie(qtd,
            labels=list(idx),
            #colors=["#20257c", "#424ad1", "#6a8ee8", "#66bbe2", "#66dee2"],
            colors = lst_cores,
            labeldistance=1.1,
            #explode=[0.08, 0.05],
            #explode=[0, 0, .1, .2, .4],
            wedgeprops={"ec": "k"},
            textprops={"fontsize": 15},
            )
    plt.axis("equal")
    plt.title("Quantidade de profissionais que preencheram o questionario por UF")
    plt.legend()
    col2.pyplot(grafico01)
    col2.pyplot(grafico02)
    #col2.plotly(grafico)

with col2:

    if tipo_analise == "Mudou de Estado x Gênero":
        grafico_genero_mudouEstado()
    elif tipo_analise == "Salário por gênero":
        grafico_genero_salario()
    elif tipo_analise == "Porcentagem por gênero":
        grafico_genero_porcentagem()
    elif tipo_analise == "Gênero por Estado":
        grafico_uf_genero()
    elif tipo_analise == "Gênero por Região":
        grafico_regiao_genero()
    elif tipo_analise == "Gênero por Escolaridade":
        grafico_escolaridade_genero()
    elif tipo_analise == "Mapeamento por Estado":
        grafico_mapa_uf()
    elif tipo_analise == "Faixa Etária":
        grafico_faixaIdade()
    elif tipo_analise == "Responderam questionário por Estado":
        grafico_questionario_uf()


    # if(eixo_X == 'Genero'):
    #     if (eixo_Y == 'Mudou de Estado?'):
    #         grafico_genero_mudouEstado()
    #     elif (eixo_Y == 'Faixa salarial'):
    #         grafico_genero_salario()
    #     elif (eixo_Y == 'Genero'):
    #         grafico_genero_porcentagem()
    # elif (eixo_Y == 'Genero'):
    #     if (eixo_X == 'uf onde mora'):
    #         grafico_uf_genero()
    #     elif (eixo_X == 'Regiao onde mora'):
    #         grafico_regiao_genero()
    #     elif (eixo_X == 'Faixa salarial'):
    #         grafico_salario_genero()
    #     elif (eixo_X == 'Nivel de Ensino'):
    #         grafico_escolaridade_genero()
    # elif(eixo_X == 'uf onde mora' and eixo_Y == 'uf onde mora'):
    #     grafico_mapa_uf()
    # elif (eixo_X == 'Faixa idade' and eixo_Y == 'Faixa idade'):
    #     grafico_faixaIdade()
    # elif (eixo_X == 'Idade' and eixo_Y == 'Idade'):
    #     grafico_idade()
    # elif (eixo_X == 'Regiao onde mora' and eixo_Y == 'Regiao onde mora'):
    #     grafico_questionario_uf()
