from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Guardamos en variables los nombres de las columnas para usarlas después
cols_numericas = ['age', 'avg_glucose_level', 'bmi']
cols_texto = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
              'work_type', 'Residence_type', 'smoking_status']

def separar_xy(df):
    # X son los datos del paciente, y es la respuesta (si tuvo ACV o no)
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    return X, y

def armar_preprocesador():
    # Reglas para las columnas numéricas: llenar vacíos con la mediana y escalar los números
    paso_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Reglas para las columnas de texto: llenar vacíos con el más común y pasarlos a números (OneHot)
    paso_texto = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Juntamos ambas reglas
    preprocesador = ColumnTransformer([
        ('num', paso_num, cols_numericas),
        ('cat', paso_texto, cols_texto)
    ])

    return preprocesador