"""
Prueba de inferencia para todos los modelos entrenados
"""
import time
import pickle
from utils import cargar_datos, evaluar_modelo


def probar_modelo(nombre_archivo, nombre_modelo):
    """
    nombre_archivo: ruta del archivo pickle del modelo
    nombre_modelo: nombre descriptivo para mostrar
    """
    _, X_test, _, y_test = cargar_datos()
    
    with open(nombre_archivo, 'rb') as f:
        modelo = pickle.load(f)
    
    inicio = time.time()
    precision = evaluar_modelo(modelo, X_test, y_test)
    tiempo_inferencia = time.time() - inicio
    
    print("{} - Tiempo inferencia: {:.6f}s | Precision: {:.4f}".format(
        nombre_modelo, tiempo_inferencia, precision))


def main():
    modelos = [
        ('models/knn.pkl', 'KNN'),
        ('models/svm.pkl', 'SVM'),
        ('models/logreg.pkl', 'LogReg'),
        ('models/mlp.pkl', 'MLP'),
        ('models/gboost.pkl', 'Gradient Boosting'),
        ('models/rforest.pkl', 'Random Forest')
    ]
    
    print("=== Resultados de Inferencia ===\n")
    
    for archivo, nombre in modelos:
        probar_modelo(archivo, nombre)


if __name__ == '__main__':
    main()