"""
Funciones utilitarias para carga de datos y evaluacion
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import config


def cargar_datos():
    """
    Retorna: X_train, X_test, y_train, y_test (normalizados)
    """
    # Leer CSV del dataset configurado
    archivo = '{}.csv'.format(config.DATASET_NAME)
    
    # Cargar datos (skip_header=1 ignora la primera linea)
    datos = np.genfromtxt(archivo, delimiter=',', skip_header=1)
    
    # Separar features y target
    X = datos[:, :-1]  # Todas las columnas excepto la ultima
    y = datos[:, -1].astype(int)  # Ultima columna como enteros
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def evaluar_modelo(modelo, X_test, y_test):
    """
    modelo: modelo entrenado con metodo predict
    X_test: datos de prueba
    y_test: etiquetas reales
    Retorna: precision (accuracy)
    """
    y_pred = modelo.predict(X_test)
    return accuracy_score(y_test, y_pred)