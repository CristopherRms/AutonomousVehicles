"""
Entrenamiento del modelo Multi-Layer Perceptron
"""
import time
import pickle
from sklearn.neural_network import MLPClassifier
from utils import cargar_datos, evaluar_modelo
import config


def entrenar_mlp():
    X_train, X_test, y_train, y_test = cargar_datos()
    
    modelo = MLPClassifier(hidden_layer_sizes=(8,8), max_iter=100, 
                          random_state=config.RANDOM_STATE)
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    precision = evaluar_modelo(modelo, X_test, y_test)
    
    with open('models/mlp.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    print("MLP - Tiempo entrenamiento: {:.6f}s | Precision: {:.4f}".format(
        tiempo_entrenamiento, precision))


if __name__ == '__main__':
    entrenar_mlp()