# IVF+HNSW

Proyecto en C++ que busca, entre miles de vectores, cuáles son los más parecidos a uno dado (búsqueda de vecinos más cercanos). En vez de comparar contra todos los vectores uno por uno (lento), usa estructuras que aceleran la búsqueda a costa de perder un poco de precisión.

## ¿Qué hace el programa?

`main.cpp` :

1. Genera datos aleatorios (10 000 vectores) y consultas (200 vectores).
2. Calcula la respuesta "correcta" comparando fuerza bruta (revisa todo, es lento pero exacto).
3. Prueba dos formas más rápidas de buscar y compara qué tan buenas son:
   - **HNSW**: arma un grafo que conecta vectores parecidos entre sí, y lo recorre como un mapa para llegar rápido a los vecinos más cercanos.
   - **IVF+HNSW**: primero agrupa todos los vectores en "zonas" (clusters), y usa HNSW solo para elegir a qué zonas mirar. Luego busca solo dentro de esas zonas, en vez de revisar todo. Es más rápido, pero un poco menos preciso.
4. Al final imprime, para cada método: cuánto tardó y qué tan preciso fue (**Recall@10** = de los 10 vecinos reales, cuántos encontró).

El cálculo se acelera con **OpenMP** (usa varios núcleos del CPU al mismo tiempo).

## Archivos del proyecto

```
main.cpp                    # demo: corre las 3 búsquedas y compara resultados
load_fvecs.h                 # para leer datasets en formato .fvecs / .ivecs (opcional)
ivf_hnsw/
  hnsw.h / hnsw.cpp           # implementación del índice HNSW
  ivf_hnsw.h / ivf_hnsw.cpp    # implementación del índice IVF+HNSW (usa HNSW por dentro)
CMakeLists.txt                # configuración para compilar el proyecto
```

## Cómo compilarlo

Necesitas tener instalado:
- CMake (versión 3.14 o más nueva)
- Un compilador de C++17 con soporte de OpenMP (en este proyecto se usa MSVC de Visual Studio)

Pasos:

```powershell
cmake -S . -B build
cmake --build build
```

> **Nota para Windows con MSVC:** estos dos comandos deben correrse desde una terminal con el entorno de Visual Studio cargado (por ejemplo, abriendo "Developer PowerShell for VS" desde el menú de inicio), porque `cmake` usa el compilador `cl.exe` de Visual Studio.

## Cómo ejecutarlo

Una vez compilado, corre:

```powershell
.\build\demo.exe
```

Esto no necesita el entorno de Visual Studio, solo el `.exe` ya compilado.

### Salida esperada (ejemplo)

```
=== IVF+HNSW demo ===
dim=100  n_data=10000  n_queries=200  k=10

--- Brute-Force (ground truth) ---
  query: 29.2 ms

--- HNSW  (M=16, ef_construction=200, ef=50) ---
  build : 13566.2 ms
  query : 12.5 ms  |  Recall@10 = 0.812

--- IVF+HNSW  (nlist=50, nprobe=8, M=16, ef_construction=200) ---
  train : 117.596 ms
  add   : 73.684 ms  (10000 vectores)
  query : 5.198 ms  |  Recall@10 = 0.420
```


