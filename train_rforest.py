"""
Entrenamiento del modelo Random Forest
"""
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from utils import cargar_datos, evaluar_modelo
import config


def entrenar_rforest():
    X_train, X_test, y_train, y_test = cargar_datos()
    
    modelo = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=config.RANDOM_STATE)
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    precision = evaluar_modelo(modelo, X_test, y_test)
    
    with open('models/rforest.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    print("Random Forest - Tiempo entrenamiento: {:.6f}s | Precision: {:.4f}".format(
        tiempo_entrenamiento, precision))


if __name__ == '__main__':
    entrenar_rforest()