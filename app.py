from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# os.chdir(os.path.dirname(__file__))

import subprocess

# Ejecuta el comando "pip list" y captura la salida
resultado = subprocess.run(['pip', 'list'], capture_output=True, text=True)

# Imprime la lista de paquetes
print(resultado.stdout)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return render_template("index.html")

# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/formulario_predict", methods=["GET"])
def formulario_predict():
    campos = [
        {"name": "fixed_acidity", "label": "Fixed Acidity", "default": 6.6},
        {"name": "volatile_acidity", "label": "Volatile Acidity", "default": 0.16},
        {"name": "citric_acid", "label": "Citric Acid", "default": 0.3},
        {"name": "residual_sugar", "label": "Residual Sugar", "default": 1.6},
        {"name": "chlorides", "label": "Chlorides", "default": 0.034},
        {"name": "free_sulfur_dioxide", "label": "Free Sulfur Dioxide", "default": 15.0},
        {"name": "total_sulfur_dioxide", "label": "Total Sulfur Dioxide", "default": 78.0},
        {"name": "density", "label": "Density", "default": 0.992},
        {"name": "pH", "label": "pH", "default": 3.38},
        {"name": "sulphates", "label": "Sulphates", "default": 0.44},
        {"name": "alcohol", "label": "Alcohol (si lo desconoces, pon 0)", "default": 0},
        {"name": "quality", "label": "Quality (0-10)", "default": 6},
        {"name": "class_", "label": "Class (white/red)", "default": "white"}
    ]
    return render_template("formulario_predict.html", fields=campos)

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=["GET"])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open('modelo_pipeline_reg.pkl', 'rb') as f:
        model = pickle.load(f)

    fixed_acidity = request.args.get('fixed_acidity', None)
    volatile_acidity = request.args.get('volatile_acidity', None)
    citric_acid = request.args.get('citric_acid', None)
    residual_sugar = request.args.get('residual_sugar', None)
    chlorides = request.args.get('chlorides', None)
    free_sulfur_dioxide = request.args.get('free_sulfur_dioxide', None)
    total_sulfur_dioxide = request.args.get("total_sulfur_dioxide", None)
    density = request.args.get('density', None)
    pH = request.args.get('pH', None)
    sulphates = request.args.get('sulphates', None)
    alcohol = request.args.get('alcohol', None)
    quality = request.args.get('quality', None)
    class_ = request.args.get('class_', None)


    print(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality, class_)
    print(type(fixed_acidity))

    if (fixed_acidity is None or
        volatile_acidity is None or
        citric_acid is None or
        residual_sugar is None or
        chlorides is None or
        free_sulfur_dioxide is None or
        total_sulfur_dioxide is None or
        density is None or
        pH is None or
        sulphates is None or
        alcohol is None or
        quality is None or
        class_ is None):
        return "Args empty, not enough data to predict"
    
    else:
    
        input_data = pd.DataFrame([{
            'fixed_acidity': float(fixed_acidity),
            'volatile_acidity': float(volatile_acidity),
            'citric_acid': float(citric_acid),
            'residual_sugar': float(residual_sugar),
            'chlorides': float(chlorides),
            'free_sulfur_dioxide': float(free_sulfur_dioxide),
            'total_sulfur_dioxide': float(total_sulfur_dioxide),
            'density': float(density),
            'pH': float(pH),
            'sulphates': float(sulphates),
            'alcohol': float(alcohol),
            'quality': int(quality),
            'class_': class_
        }])

        prediction = model.predict(input_data)
        
    
    # return jsonify({'prediction': float(prediction[0])}) 
    return jsonify({'prediction': round(float(prediction[0]), 2)})


# NUEVA RUTA HTML PARA EL FORMULARIO DE REENTRENAMIENTO
@app.route("/api/v1/retrain_page", methods=["GET"])
def retrain_page():
    return render_template("retrain.html")


# Enruta la funcion al endpoint /api/v1/retrain
@app.route("/api/v1/retrain/", methods=["GET"])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/wines_train.csv"):
        data = pd.read_csv('data/wines_train.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['alcohol']),
                                                        data['alcohol'],
                                                        test_size = 0.20,
                                                        random_state=42)
        
        
        features_num_reg_1 = ['density','residual_sugar','total_sulfur_dioxide','chlorides','free_sulfur_dioxide','pH','fixed_acidity']
        features_cat_reg = ['class_','quality']
        
        columns_to_keep_reg = features_num_reg_1 + features_cat_reg

        columns_to_exclude_reg = [col for col in X_train.columns if col not in columns_to_keep_reg]
                     
        cat_pipeline = Pipeline([("Impute_Mode", SimpleImputer(strategy="most_frequent")),  # Imputación con la moda
                                 ("OHEncoder", OneHotEncoder(handle_unknown='ignore'))  # Manejar categorías desconocidas
                                 ])

        logaritmica = FunctionTransformer(np.log1p, feature_names_out="one-to-one") 
        # Esto le indica al Pipeline que el número de características no cambia y que puede usar los nombres originales.

        num_pipeline = Pipeline(
            [("Impute_Mean", SimpleImputer(strategy = "mean")), # prevision que en el futuro lleguen datos faltantes
            ("logaritmo", logaritmica),
            ("SScaler", StandardScaler()),
            ]   )

        imputer_step_reg = ColumnTransformer(
            [("Process_Numeric", num_pipeline,features_num_reg_1), # feature_numericas seleccionadas para clasificación
            ("Process_Categorical", cat_pipeline, features_cat_reg), # feature_categoriacas seleccionadas para regresión
            ("Exclude", "drop", columns_to_exclude_reg)
            ], remainder = "passthrough"
            )

        pipe_missings_reg = Pipeline([("first_stage", imputer_step_reg)])
        
        XGB_pipeline = Pipeline(
             [("Preprocesado", pipe_missings_reg),  
              ("Modelo", XGBRegressor(learning_rate= 0.1, max_depth= 8, n_estimators= 400))  # Modelo de boosting basado en XGBoost
              ])

        resultado_reg = cross_val_score(XGB_pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_percentage_error")
        XGB_pipeline.fit(X_train,y_train)
        y_pred=XGB_pipeline.predict(X_test)
        mape_test=mean_absolute_percentage_error(y_pred=y_pred,y_true=y_test)

        with open('modelo_pipeline_reg.pkl', 'wb') as archivo:
            pickle.dump(XGB_pipeline, archivo)
            
        return f"Model retrained. New evaluation metric MAPE_train: {np.mean(-resultado_reg) * 100:.2f}% y MAPE_test: {mape_test * 100:.2f}%"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


