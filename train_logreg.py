"""
Entrenamiento del modelo Logistic Regression
"""
import time
import pickle
from sklearn.linear_model import LogisticRegression
from utils import cargar_datos, evaluar_modelo


def entrenar_logreg():
    X_train, X_test, y_train, y_test = cargar_datos()
    
    modelo = LogisticRegression(solver='lbfgs', max_iter=500, 
                                C=1.0, tol=0.001)
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    precision = evaluar_modelo(modelo, X_test, y_test)
    
    with open('models/logreg.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    print("LogReg - Tiempo entrenamiento: {:.6f}s | Precision: {:.4f}".format(
        tiempo_entrenamiento, precision))


if __name__ == '__main__':
    entrenar_logreg()