import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score,
                             precision_recall_curve)

def evaluar_modelo(modelo, X_test, y_test):
    # Creamos la carpeta outputs si no existe
    os.makedirs("outputs", exist_ok=True)
    
    # Sacamos las probabilidades de que un paciente tenga ACV
    y_proba = modelo.predict_proba(X_test)[:, 1]

    # Calculamos el ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Buscamos el mejor umbral (corte) para no dejar pasar a los enfermos
    prec, rec, umbrales = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    umbral_optimo = umbrales[np.argmax(f1_scores[:-1])]
    print(f"Umbral óptimo: {umbral_optimo:.2f}")

    # Hacemos las predicciones finales usando ese umbral
    y_pred = (y_proba >= umbral_optimo).astype(int)

    # Mostramos el reporte en texto
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Sano (0)', 'ACV (1)']))

    # --- GRÁFICA 1: MATRIZ DE CONFUSIÓN ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Sano', 'ACV'], yticklabels=['Sano', 'ACV'])
    axes[0].set_title(f'Matriz de Confusión (Umbral: {umbral_optimo:.2f})')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicción')

    # --- GRÁFICA 2: CURVA ROC ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color='red', label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title('Curva ROC')
    axes[1].set_xlabel('Falsos Positivos')
    axes[1].set_ylabel('Verdaderos Positivos')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/graficas_evaluacion.png")
    plt.show()

def graficar_importancias(modelo, cols_numericas, cols_texto):
    # Sacamos los nombres de las columnas transformadas
    pre = modelo.named_steps['preprocesador']
    ohe = pre.named_transformers_['cat'].named_steps['onehot']
    nombres_columnas = list(cols_numericas) + list(ohe.get_feature_names_out(cols_texto))

    # Sacamos la importancia que le dio el Random Forest
    importancias = modelo.named_steps['modelo'].feature_importances_

    # Creamos una tabla y la ordenamos
    tabla = pd.DataFrame({'Variable': nombres_columnas, 'Importancia': importancias})
    tabla = tabla.sort_values(by='Importancia', ascending=False).head(10)

    # Graficamos
    plt.figure(figsize=(8, 5))
    sns.barplot(data=tabla, x='Importancia', y='Variable', palette='viridis')
    plt.title('Top 10 variables más importantes')
    plt.tight_layout()
    plt.savefig("outputs/importancias.png")
    plt.show()

def guardar_modelo(modelo):
    # Guardamos el modelo entrenado para usarlo después
    joblib.dump(modelo, "outputs/modelo_acv.joblib")
