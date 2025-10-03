"""
Entrenamiento del modelo SVM
"""
import time
import pickle
from sklearn.svm import SVC
from utils import cargar_datos, evaluar_modelo


def entrenar_svm():
    X_train, X_test, y_train, y_test = cargar_datos()
    
    modelo = SVC(kernel='rbf', C=1)
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    precision = evaluar_modelo(modelo, X_test, y_test)
    
    with open('models/svm.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    print("SVM - Tiempo entrenamiento: {:.6f}s | Precision: {:.4f}".format(
        tiempo_entrenamiento, precision))


if __name__ == '__main__':
    entrenar_svm()