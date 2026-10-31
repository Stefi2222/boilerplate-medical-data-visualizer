import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de los pacientes
df = pd.read_csv('medical_examination.csv')

# Calcular el IMC y determinar si existe sobrepeso
df['overweight'] = (
    df['weight'] / (df['height'] / 100) ** 2 > 25
).astype(int)

# Normalizar colesterol y glucosa
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# Crear gráfico de categorías
def draw_cat_plot():

    # Organizar los datos
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=[
            'cholesterol',
            'gluc',
            'smoke',
            'alco',
            'active',
            'overweight'
        ]
    )

    # Agrupar y contar los registros
    df_cat = (
        df_cat.groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Crear el gráfico
    chart = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    fig = chart.fig

    # Guardar imagen
    fig.savefig('catplot.png')

    return fig


# Crear mapa de calor de correlaciones
def draw_heat_map():

    # Limpiar registros inconsistentes
    df_heat = df.loc[
        (df['ap_lo'] <= df['ap_hi'])
        &
        (df['height'] >= df['height'].quantile(0.025))
        &
        (df['height'] <= df['height'].quantile(0.975))
        &
        (df['weight'] >= df['weight'].quantile(0.025))
        &
        (df['weight'] <= df['weight'].quantile(0.975))
    ].copy()

    # Calcular correlaciones
    corr = df_heat.corr()

    # Ocultar la mitad superior
    mask = np.triu(
        np.ones_like(corr, dtype=bool)
    )

    # Crear el mapa de calor
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        ax=ax
    )

    # Guardar imagen
    fig.savefig('heatmap.png')

    return fig
