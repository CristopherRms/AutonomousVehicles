"""
Entrenamiento del modelo KNN
"""
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from utils import cargar_datos, evaluar_modelo
import config


def entrenar_knn():
    X_train, X_test, y_train, y_test = cargar_datos()
    
    modelo = KNeighborsClassifier(n_neighbors=3)
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    precision = evaluar_modelo(modelo, X_test, y_test)
    
    with open('models/knn.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    print("KNN - Tiempo entrenamiento: {:.6f}s | Precision: {:.4f}".format(
        tiempo_entrenamiento, precision))


if __name__ == '__main__':
    entrenar_knn()