from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import Lasso
import numpy as np
import seaborn as sns
from flask import render_template

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Enruta la landing page (endpoint /)
#HELLO DE RODRIGO, HAY QUE ADAPTARLO A NUESTRO TRABAJO (COMENTADO PARA PROBAR LANDPAGE CHATGPT)
'''
@app.route("/", methods=["GET"])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return """
    <h1>Bienvenido a la API del modelo alcohol en tu vino, diseñado por el grupo Vinos formado por: Rommel, Rodrigo, Guillermo y Jose Luis</h1>
    <p>Opciones disponibles:</p>
    <ul>
        <li><strong>/</strong> - Página inicial.</li>
        <li><strong>/api/v1/predict</strong> - Endpoint para realizar predicciones. <br> Usa parámetros [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality, class_] en la URL para predecir.</li>
        <li><strong>/api/v1/retrain</strong> - Endpoint para reentrenar el modelo con datos nuevos. <br> Busca automáticamente el archivo 'Advertising_new.csv' en la carpeta 'data'.</li>
    </ul>
    <p>Para más información, accede a cada endpoint según corresponda.</p>
    """
'''
#LANDPAGE CHATGPT PREDICCIÓN EN LANDPAGE
@app.route("/", methods=["GET"])
def hello():
    return """
    <h1>Bienvenido a la API del modelo 'Alcohol en tu vino' 🍷</h1>
    <p>Introduce los valores del vino para predecir:</p>
    
    <form id="predictionForm">
        <label>Fixed Acidity: <input name="fixed_acidity" step="any"></label><br>
        <label>Volatile Acidity: <input name="volatile_acidity" step="any"></label><br>
        <label>Citric Acid: <input name="citric_acid" step="any"></label><br>
        <label>Residual Sugar: <input name="residual_sugar" step="any"></label><br>
        <label>Chlorides: <input name="chlorides" step="any"></label><br>
        <label>Free Sulfur Dioxide: <input name="free_sulfur_dioxide" step="any"></label><br>
        <label>Total Sulfur Dioxide: <input name="total_sulfur_dioxide" step="any"></label><br>
        <label>Density: <input name="density" step="any"></label><br>
        <label>pH: <input name="pH" step="any"></label><br>
        <label>Sulphates: <input name="sulphates" step="any"></label><br>
        <label>Alcohol: <input name="alcohol" step="any"></label><br>
        <label>Quality (0-10): <input name="quality" step="1"></label><br>
        <label>Class (white/red): <input name="class_" value="white"></label><br><br>
        <button type="submit">Predecir</button>
    </form>

    <h3>Resultado: <span id="predictionResult">---</span></h3>

    <script>
    document.getElementById("predictionForm").addEventListener("submit", function(event) {
        event.preventDefault();
        const formData = new FormData(event.target);
        const params = new URLSearchParams();
        for (const [key, value] of formData.entries()) {
            params.append(key, value);
        }
        fetch('/api/v1/predict?' + params.toString())
            .then(response => response.json())
            .then(data => {
                document.getElementById("predictionResult").innerText = data.prediction;
            })
            .catch(error => {
                document.getElementById("predictionResult").innerText = "Error: " + error;
            });
    });
    </script>
    """

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



# Enruta la funcion al endpoint /api/v1/retrain
@app.route("/api/v1/retrain/", methods=["GET"])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(round(mape*100,2))}%"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    app.run(debug=True)
