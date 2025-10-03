"""
Benchmark para LEGO EV3 - Mide CPU, memoria y I/O
"""
import time
import os
import tempfile


def benchmark_cpu(iteraciones=100000):
    """
    iteraciones: numero de operaciones matematicas
    Retorna: tiempo en segundos
    """
    inicio = time.time()
    resultado = 0
    for i in range(iteraciones):
        resultado += i ** 2
    return time.time() - inicio


def benchmark_memoria(tamano_mb=10):
    """
    tamano_mb: MB de memoria a alocar
    Retorna: tiempo en segundos
    """
    inicio = time.time()
    datos = [0] * (tamano_mb * 1024 * 128)
    datos[0] = 1
    datos[-1] = 1
    return time.time() - inicio


def benchmark_io(iteraciones=1000):
    """
    iteraciones: numero de escrituras/lecturas
    Retorna: tiempo en segundos
    """
    archivo = os.path.join(tempfile.gettempdir(), 'benchmark_test.txt')
    inicio = time.time()
    
    with open(archivo, 'w') as f:
        for i in range(iteraciones):
            f.write("Linea de prueba {}\n".format(i))
    
    with open(archivo, 'r') as f:
        contenido = f.read()
    
    os.remove(archivo)
    return time.time() - inicio


def benchmark_bucles(iteraciones=1000000):
    """
    iteraciones: operaciones simples en bucle
    Retorna: operaciones por segundo
    """
    inicio = time.time()
    contador = 0
    for i in range(iteraciones):
        contador += 1
    tiempo = time.time() - inicio
    return iteraciones / tiempo if tiempo > 0 else 0


def main():
    print("=== Benchmark LEGO EV3 ===\n")
    
    print("1. CPU (operaciones matematicas)...")
    t_cpu = benchmark_cpu(100000)
    print("   Tiempo: {:.4f}s\n".format(t_cpu))
    
    print("2. Memoria (allocacion 10MB)...")
    t_mem = benchmark_memoria(10)
    print("   Tiempo: {:.4f}s\n".format(t_mem))
    
    print("3. I/O Disco (1000 escrituras/lecturas)...")
    t_io = benchmark_io(1000)
    print("   Tiempo: {:.4f}s\n".format(t_io))
    
    print("4. Velocidad de bucles...")
    ops = benchmark_bucles(1000000)
    print("   {:.0f} ops/segundo\n".format(ops))
    
    print("=== Resumen ===")
    print("CPU: {:.4f}s | Memoria: {:.4f}s | I/O: {:.4f}s".format(
        t_cpu, t_mem, t_io))
    
    # Score simple
    score = 1000 / (t_cpu + t_mem + t_io)
    print("Score total: {:.2f} puntos".format(score))


if __name__ == '__main__':
    main()