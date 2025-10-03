"""
Visualizacion de predicciones de todos los modelos
"""
import pickle
from utils import cargar_datos
from sklearn.datasets import load_wine


def visualizar_predicciones(n_muestras=10):
    """
    n_muestras: numero de muestras a mostrar por modelo
    """
    _, X_test, _, y_test = cargar_datos()
    wine = load_wine()
    nombres_clases = wine.target_names
    nombres_features = wine.feature_names

    modelos = [
        ('models/knn.pkl', 'KNN'),
        ('models/svm.pkl', 'SVM'),
        ('models/logreg.pkl', 'LogReg'),
        ('models/mlp.pkl', 'MLP'),
        ('models/gboost.pkl', 'Gradient Boosting'),
        ('models/rforest.pkl', 'Random Forest')
    ]
    
    X_muestra = X_test[:n_muestras]
    y_muestra = y_test[:n_muestras]
    
    print("=== Visualizacion de Predicciones ===\n")
    print("Mostrando {} muestras del conjunto de prueba\n".format(n_muestras))
    
    for i, (features, real) in enumerate(zip(X_muestra, y_muestra)):
        print("-" * 70)
        print("Muestra {}: [{}]".format(
            i+1, 
            ', '.join(["{:.1f}".format(f) for f in features])
        ))
        print("Clase Real: {}".format(nombres_clases[real]))
        print()
        
        for archivo, nombre in modelos:
            with open(archivo, 'rb') as f:
                modelo = pickle.load(f)
            
            prediccion = modelo.predict([features])[0]
            correcto = "OK" if prediccion == real else "ERROR"
            
            print("  {:<20} -> {} [{}]".format(
                nombre,
                nombres_clases[prediccion],
                correcto
            ))
        
        print()


if __name__ == '__main__':
    visualizar_predicciones(10)