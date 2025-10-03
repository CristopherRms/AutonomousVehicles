"""
Entrenamiento del modelo Gradient Boosting
"""
import time
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from utils import cargar_datos, evaluar_modelo
import config


def entrenar_gboost():
    X_train, X_test, y_train, y_test = cargar_datos()

    modelo = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3,
                                       random_state=config.RANDOM_STATE)
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    precision = evaluar_modelo(modelo, X_test, y_test)
    
    with open('models/gboost.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    print("Gradient Boosting - Tiempo entrenamiento: {:.6f}s | Precision: {:.4f}".format(
        tiempo_entrenamiento, precision))


if __name__ == '__main__':
    entrenar_gboost()