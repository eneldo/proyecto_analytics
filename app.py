# Importa Flask y la función render_template para servir archivos HTML
from flask import Flask, render_template, url_for

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import plotly.io as pio
from pymongo import MongoClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
import base64
import io
# Crea una instancia de la aplicación Flask
app = Flask(__name__)


# Función para obtener el DataFrame desde MongoDB
def get_data_from_mongo():
    client = MongoClient("mongodb+srv://eltallerdevaner23:ZI7rjJJ6B0rn1MXE@cluster0.nfyfcuw.mongodb.net/")
    db = client["Curso_data"]
    collection = db["Global_cybersecurity"]
    data = collection.find()
    df = pd.json_normalize(data)

    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    
    # Eliminar columna _id si existe
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    
    return df
    


    #print("Número de registros obtenidos:", len(df))  # 👈 Verifica la conexión
    #print("Columnas disponibles:", df.columns.tolist())  # 👈 Ve qué campos hay

    #return df

# Ruta principal
@app.route('/')
def index():
    df = get_data_from_mongo()
    data = df.to_dict(orient='records')  # Lista de diccionarios para la tabla
    columns = df.columns.tolist()         # Lista de nombres de columnas
    return render_template('index.html', data=data, columns=columns)
    #return render_template("index.html")



@app.route('/nosotros')
def nosotros():
    return render_template('nosotros.html')

@app.route('/resumen')
def resumen():
    return render_template('resumen.html')


# Ruta para análisis descriptivo



@app.route('/descriptivo')
def descriptivo():
    df = get_data_from_mongo()
    
    # Aquí podrías hacer análisis estadístico con df y pasar resultados al HTML
    resumen_html = df.describe(include='all').to_html(
        classes="table table-striped",
        border=0,
        justify='center'
    )

    #2. Gráfico de barras: pérdidas por país
    # ============================
    if 'Country' in df.columns and 'Financial Loss (in Million $)' in df.columns:
        df_bar = df.groupby('Country', as_index=False)['Financial Loss (in Million $)'].sum()
        df_bar = df_bar.sort_values(by='Financial Loss (in Million $)', ascending=False)

        fig_bar = px.bar(
            df_bar,
            x='Country',
            y='Financial Loss (in Million $)',
            title='Pérdidas Financieras por País',
            labels={'Financial Loss (in Million $)': 'Pérdida (Millones USD)', 'Country': 'País'},
            color='Financial Loss (in Million $)',
            color_continuous_scale='Reds'
        )
        graph_bar = pio.to_html(fig_bar, full_html=False)
    else:
        graph_bar = "<p>No se encontraron columnas de país o pérdidas financieras.</p>"


    # 3. Gráfico de torta: tipos de ataque
    # ============================
    if 'Attack Type' in df.columns:
        df_pie = df['Attack Type'].value_counts().reset_index()
        df_pie.columns = ['Attack Type', 'Count']

        fig_pie = px.pie(
            df_pie,
            names='Attack Type',
            values='Count',
            title='Distribución de Tipos de Ataque',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        graph_pie = pio.to_html(fig_pie, full_html=False)
    else:
        graph_pie = "<p>No se encontró la columna 'Attack Type'.</p>"

    return render_template("descriptivo.html", resumen=resumen_html, graph_bar=graph_bar,graph_pie=graph_pie)





# Ruta para análisis diferencial


@app.route("/diferencial")
def diferencial():
    # 1. Obtener datos desde MongoDB
    df = get_data_from_mongo()

    # 2. Validación de columnas necesarias
    if 'Country' not in df.columns or 'Financial Loss (in Million $)' not in df.columns:
        return "Faltan columnas necesarias en los datos (Country o Financial Loss)."

    # 3. Análisis diferencial: Media de pérdidas por país
    df_diff = df.groupby("Country")["Financial Loss (in Million $)"].mean().reset_index()
    df_diff = df_diff.sort_values(by="Financial Loss (in Million $)", ascending=False)

    # 4. Gráfico de barras (Top 10 países)
    fig_diff = px.bar(
        df_diff.head(10),
        x="Country",
        y="Financial Loss (in Million $)",
        title="Media de Pérdidas Financieras por País (Top 10)",
        labels={"Financial Loss (in Million $)": "Pérdida Promedio (Millones $)"},
        color="Financial Loss (in Million $)",
        text_auto=".2s",
    )
    fig_diff.update_layout(xaxis_title="País", yaxis_title="Pérdida Promedio (Millones $)")
    graph_diff = pio.to_html(fig_diff, full_html=False)

    # 5. Renderizar plantilla con gráfico
    return render_template("diferencial.html", graph_diff=graph_diff)




# Ruta para análisis exploratorio

import plotly.express as px
import plotly.io as pio
from markupsafe import Markup  


@app.route('/exploratorio')
def exploratorio():
    df = get_data_from_mongo()

    # === Gráfico 1: Correlación (solo numéricos) ===
    correlation_html = ""
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().stack().reset_index()
        corr_matrix.columns = ['var1', 'var2', 'correlation']
        fig_corr = px.imshow(numeric_df.corr(), 
                             title="Matriz de Correlación",
                             color_continuous_scale='RdBu_r', 
                             zmin=-1, zmax=1)
        correlation_html = Markup(pio.to_html(fig_corr, full_html=False))

    # === Gráfico 2: Barras de pérdidas financieras promedio ===
    bar_html = ""
    if 'Country' in df.columns and 'Financial Loss (in Million $)' in df.columns:
        bar_data = df.groupby('Country')["Financial Loss (in Million $)"].mean().sort_values(ascending=False).head(10).reset_index()
        fig_bar = px.bar(bar_data, 
                         x='Financial Loss (in Million $)', 
                         y='Country', 
                         orientation='h', 
                         title="Top 10 países por pérdida financiera promedio",
                         color='Financial Loss (in Million $)', 
                         color_continuous_scale='viridis')
        bar_html = Markup(pio.to_html(fig_bar, full_html=False))

    # === Gráfico 3: Histograma de usuarios afectados ===
    hist_html = ""
    if 'Number of Affected Users' in df.columns:
        fig_hist = px.histogram(df, 
                                x="Number of Affected Users", 
                                nbins=30, 
                                title="Distribución de usuarios afectados por incidente")
        hist_html = Markup(pio.to_html(fig_hist, full_html=False))

    return render_template("exploratorio.html",
                           correlation_html=correlation_html,
                           bar_html=bar_html,
                           hist_html=hist_html)

# fin # Ruta para análisis exploratorio


# Inicio de la ruta predictivo

# # vaner hasta aqui esta funcionando bien 



def plot_correlation_heatmap(df, features):
    plt.figure(figsize=(8,6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa de Correlación")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_feature_importance(importance_df):
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Importancia de Variables en Random Forest',
        labels={'importance': 'Importancia', 'feature': 'Variable'}
    )
    return pio.to_html(fig, full_html=False)

@app.route('/predictivo')
def predictivo():
    df = get_data_from_mongo()

    target = 'Financial Loss (in Million $)'
    features = ['Year', 'Number of Affected Users']

    # Convertir a numérico y limpiar NaNs
    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=features + [target]).copy()

    X = df_clean[features]
    y = df_clean[target]

    # Mapa de correlación
    corr_img = plot_correlation_heatmap(df_clean, features + [target])

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo Regresión Polinómica grado 2
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_pred_poly = poly_model.predict(X_test)
    r2_poly = r2_score(y_test, y_pred_poly)

    # Modelo Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Importancia variables Random Forest
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=True)

    graph_importance = plot_feature_importance(imp_df)

    # Gráficos comparación Actual vs Predicho (Random Forest)
    df_pred_comp = pd.DataFrame({'Actual': y_test, 'Predicho_RF': y_pred_rf, 'Predicho_Poly': y_pred_poly})

    fig_pred_comp_rf = px.scatter(
        df_pred_comp,
        x='Actual',
        y='Predicho_RF',
        trendline='ols',
        title='Actual vs Predicho - Random Forest'
    )
    graph_pred_comp_rf = pio.to_html(fig_pred_comp_rf, full_html=False)

    # Gráfico comparación Actual vs Predicho (Regresión Polinómica)
    fig_pred_comp_poly = px.scatter(
        df_pred_comp,
        x='Actual',
        y='Predicho_Poly',
        trendline='ols',
        title='Actual vs Predicho - Regresión Polinómica'
    )
    graph_pred_comp_poly = pio.to_html(fig_pred_comp_poly, full_html=False)

    return render_template('predictivo.html',
        corr_img=corr_img,
        r2_poly=round(r2_poly, 4),
        r2_rf=round(r2_rf, 4),
        graph_importance=graph_importance,
        graph_pred_comp_rf=graph_pred_comp_rf,
        graph_pred_comp_poly=graph_pred_comp_poly
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render define el puerto
    app.run(host='0.0.0.0', port=port)




"""
@app.route('/predictivo')
def predictivo():
    # 1. Obtener datos desde Mongo
    df = get_data_from_mongo()

    # Copia para evitar modificar el original
    data = df.copy()

    # 2. Limpieza y conversión
    for col in ['Year', 'Financial Loss (in Million $)']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    for col in ['Country', 'Attack Type']:
        if col in data.columns:
            data[col] = data[col].astype(str)

    # Verificación de columnas requeridas
    required_cols = ['Year', 'Financial Loss (in Million $)', 'Country', 'Attack Type']
    for col in required_cols:
        if col not in data.columns:
            return f"Error: falta la columna {col} en los datos."

    # 3. Preparar X e y
    X = data[['Year', 'Country', 'Attack Type']]
    y = data['Financial Loss (in Million $)']

    # Eliminar filas con target NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    # 4. Pipeline preprocesamiento
    numeric_features = ['Year']
    categorical_features = ['Country', 'Attack Type']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # 5. Pipeline completo con regresión lineal
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # 6. Entrenar modelo
    model.fit(X, y)

    # 7. Predecir y calcular R²
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # 8. Gráfico Actual vs Predicho
    fig = px.scatter(x=y, y=y_pred,
                     labels={'x': 'Valor Real (Pérdida Financiera)', 'y': 'Valor Predicho'},
                     title=f'Regresión Lineal: Actual vs Predicho (R² = {r2:.3f})')
    fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                  line=dict(color='Red', dash='dash'))

    graph_html = pio.to_html(fig, full_html=False)

    # 9. Predicción para el año 2025 con país y ataque más frecuentes
    most_freq_country = data['Country'].mode()[0]
    most_freq_attack = data['Attack Type'].mode()[0]

    input_2025 = pd.DataFrame({
        'Year': [2025],
        'Country': [most_freq_country],
        'Attack Type': [most_freq_attack]
    })

    pred_2025 = model.predict(input_2025)[0]

    # 10. Renderizar template
    return render_template('predictivo.html',
                           graph=graph_html,
                           r2=r2,
                           pred_2025=pred_2025,
                           country_2025=most_freq_country,
                           attack_2025=most_freq_attack)

"""


# ya esta bloquado aahora voy a ingresar el otro codigo. 



"""
if __name__ == '__main__':
    app.run(debug=True)"""