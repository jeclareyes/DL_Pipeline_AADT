"""

Este pipeline deberá coordinar lo siguiente:

0.1 Qué case study se trabajará (SiouxFalls, Barcelona, etc.) (el caso de estudio se coge de raw o de external)
0.2 Qué modelo se empleará (para entrenamiento o evaluación se podrá elegir entre varios modelos implementados)
0.3 Según el modelo seleccionado: Parámetros de entrenamiento
0.4 Qué etapas de la evaluación se ejecutarán (tablas, gráficos, comparativas, etc.) (también, se podrá llamar
    algoritmos de traffic assignment clásicos para comparar las ground truth con los resultados del modelo en los casos
    que esta exista)
0.5 Qué matriz OD se usa (¿existe solo una?, ¿existen varios días de registro?, ¿si es así, qué días se van a usar para
    entrenamiento y cuáles para evaluación?, ¿según su fecha DD-MM-AAAA?
1.0 Carga de datos (se cogen los archivos de raw o external según el case study) (en el caso de que sea external, hay que
    hacer un preprocesamiento previo para dejar los datos listos)
1.1 Preprocesamiento de datos (En caso de que los datos vengan de la carpeta externa habrá que preprocesarlos y dejarlos
    en la carpeta data/interim
1.2 Procesamiento de datos (según el modelo seleccionado y según la forma en que se tratará la matriz OD se procesan
    los datos para que queden listo para que el modelo los digiera. Estos se toman de la carpeta data/interim si son
    viniendo de external o de data/raw si son viniendo de raw y se dejan en data/processed)
2.0 Entrenamiento del modelo (se ejecuta el entrenamiento del modelo seleccionado con los parámetros definidos)
3.0 Evaluación del modelo (se evalúa el modelo entrenado con las métricas definidas)
4.0 Generación de reportes y gráficos (se generan los reportes y gráficos según las evaluaciones escogidas y se guardan
    en outputs/runs/run_id y en cada run_id habrá una carpeta figures, maps y models que es donde se guardan las
    figuras, mapas y modelos respectivamente)
"""