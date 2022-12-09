import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

    grafico_03 = px.bar(data_frame=df_genero_salario_ctb,
                     x=df_genero_salario_ctb.index,
                     y=df_genero_salario_ctb.columns,
                     base=df_genero_salario_ctb.index,
                     barmode="group")
    col2.plotly_chart(grafico_03)

with col2:

    if(eixo_X == 'Genero'):
        if (eixo_Y == 'Mudou de Estado?'):
            grafico_genero_mudouEstado()
        elif (eixo_Y == 'Faixa salarial'):
            grafico_genero_salario()
        else:
            grafico_genero_porcentagem()
