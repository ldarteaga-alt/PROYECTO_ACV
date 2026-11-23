from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def crear_cv():
    # Validación cruzada dividiendo los datos en 5 partes
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def comparar_modelos(X_train, y_train, preprocesador):
    # Diccionario con los 4 modelos que vamos a probar
    modelos = {
        "Regresión Logística": LogisticRegression(class_weight='balanced', max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced')
    }

    cv = crear_cv()
    print("Comparando modelos (F1-Score):")
    print("-" * 40)

    # Probar cada modelo
    for nombre, modelo in modelos.items():
        pipe = Pipeline([('preprocesador', preprocesador), ('modelo', modelo)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')
        print(f"{nombre}: {scores.mean():.4f}")

def optimizar_rf(X_train, y_train, preprocesador):
    # Armamos el modelo Random Forest
    pipe = Pipeline([
        ('preprocesador', preprocesador),
        ('modelo', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # Opciones de parámetros para probar
    parametros = {
        'modelo__n_estimators': [50, 100],
        'modelo__max_depth': [5, 10, None],
        'modelo__min_samples_split': [2, 5]
    }
    
    # GridSearchCV busca la mejor combinación
    grid = GridSearchCV(pipe, parametros, cv=crear_cv(), scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print("\nMejores parámetros para Random Forest:", grid.best_params_)
    print(f"Mejor F1-Score: {grid.best_score_:.4f}")
    
    # Retornamos el mejor modelo encontrado
    return grid.best_estimator_