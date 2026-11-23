import pandas as pd

def cargar_datos(ruta):
    # Leemos el archivo csv
    df = pd.read_csv(ruta)
    
    # Quitamos la columna 'id' porque es solo un número de paciente, no sirve para predecir
    df = df.drop('id', axis=1)
    
    # En 'gender' hay un solo caso de 'Other', lo cambiamos a 'Unknown' para que no dé error
    df['gender'] = df['gender'].replace({'Other': 'Unknown'})
    
    return df

def resumen_datos(df):
    # Imprimimos info básica para revisar cómo vienen los datos
    print("Dimensiones de la tabla:", df.shape)
    print("\nDatos vacíos (nulos):\n", df.isnull().sum())
    
    # Vemos qué porcentaje de pacientes tienen ACV
    print("\nPorcentaje de ACV:\n", df['stroke'].value_counts(normalize=True))
    print(f"Total de pacientes con ACV: {df['stroke'].sum()}")