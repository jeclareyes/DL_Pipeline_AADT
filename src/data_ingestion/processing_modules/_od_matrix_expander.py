
def add_aux_od_matrix(od_matrix, node_df):
    """Expande la matriz OD para incluir nodos auxiliares con demanda desconocida (NaN).

    Los nodos auxiliares (type='aux') no tienen demanda conocida, por lo que se añaden
    a la matriz OD con valores NaN. Esto incluye:
    - Pares aux-aux
    - Pares aux-taz
    - Pares taz-aux

    La matriz expandida será aproximadamente 108x108 (69 TAZ + ~39 aux nodes).
    """
    import numpy as np
    from scipy import sparse as sp
    import pandas as pd

    # --- 1. Verificación Inicial de Argumentos y Entorno ---

    # Si no se proporciona información de nodos, no se puede identificar TAZ y Auxiliares.
    if node_df is None or node_df.empty:
        print("   ⚠️  No hay información de nodos. No se puede expandir la matriz OD.")
        return od_matrix  # Devuelve la matriz original si no se puede expandir

    # Si no hay matriz OD cargada, no hay nada que expandir.
    if od_matrix is None:
        print("   ⚠️  No hay matriz OD cargada. No se puede expandir.")
        return None  # Retorna None ya que no hay matriz con la que trabajar

    # --- 2. Preparación Defensiva del DataFrame de Nodos ---

    # Copia defensiva: Creamos una copia explícita del DataFrame de nodos.
    # Esto evita el 'SettingWithCopyWarning' y previene efectos secundarios (mutación)
    # sobre el DataFrame original que fue pasado como argumento a la función.
    node_df = node_df.copy()

    # --- 3. Identificación de Columnas Clave ---

    # Búsqueda flexible de la columna de ID de nodo ('node', 'node_id', etc.)
    node_col = None
    for candidate in ['node', 'node_id', 'Node', 'NODE']:
        if candidate in node_df.columns:
            node_col = candidate
            break

    # Manejo de error si la columna no se encuentra y la primera columna no es adecuada
    if node_col is None:
        # En caso de no encontrar un nombre común, asumimos la primera columna
        node_col = node_df.columns[0]
        print(f"   ℹ️  Usando la primera columna '{node_col}' como ID de nodo por defecto.")

    # Búsqueda flexible de la columna de Tipo de nodo ('type', 'node_type', etc.)
    type_col = None
    for candidate in ['type', 'Type', 'TYPE', 'node_type']:
        if candidate in node_df.columns:
            type_col = candidate
            break

    # Manejo de error si no se encuentra la columna de tipo (esencial para la lógica)
    if type_col is None:
        print("   ⚠️  No se encontró columna 'type' en node_coords_df. No se puede identificar nodos auxiliares.")
        return od_matrix  # Devuelve la matriz original

    # --- 4. Extracción y Conteo de Nodos TAZ y Auxiliares ---

    # Extraer los IDs de los nodos TAZ (demanda conocida)
    taz_nodes = node_df[node_df[type_col] == 'taz'][node_col].astype(str).tolist()
    # Extraer los IDs de los nodos auxiliares (demanda desconocida)
    aux_nodes = node_df[node_df[type_col] == 'aux'][node_col].astype(str).tolist()

    # Ordenar los nodos TAZ y auxiliares. Se intenta ordenar numéricamente primero.
    def sort_key(x):
        return int(x) if x.isdigit() else x

    taz_nodes = sorted(taz_nodes, key=sort_key)
    aux_nodes = sorted(aux_nodes, key=sort_key)

    n_taz = len(taz_nodes)
    n_aux = len(aux_nodes)

    # Si no hay nodos auxiliares, la expansión no es necesaria.
    if n_aux == 0:
        print(f"   ℹ️  No se encontraron nodos auxiliares. Matriz OD permanece {n_taz}x{n_taz}")
        return od_matrix  # Devuelve la matriz original

    print(f"   🔄 Expandiendo matriz OD: {n_taz}x{n_taz} → {n_taz + n_aux}x{n_taz + n_aux}")
    print(f"      - Nodos TAZ (demanda conocida): {n_taz}")
    print(f"      - Nodos auxiliares (demanda NaN): {n_aux}")

    # --- 5. Conversión y Verificación de la Matriz OD TAZ original ---

    # Convertir la matriz OD actual a formato denso de NumPy para facilitar la inserción.
    if sp.issparse(od_matrix):
        od_dense = od_matrix.toarray()
    else:
        od_dense = np.array(od_matrix)

    # Verificar que las dimensiones de la matriz TAZ coincidan con el número de nodos TAZ encontrados.
    if od_dense.shape[0] != n_taz or od_dense.shape[1] != n_taz:
        print(f"   ⚠️  Advertencia: Matriz OD actual es {od_dense.shape}, esperado {n_taz}x{n_taz}")

        # Lógica de ajuste (opcional, dependiendo de la tolerancia a errores de datos)
        # Si la matriz es más pequeña o más grande de lo esperado, se intenta ajustar
        # a la dimensión N_TAZ.
        if od_dense.shape[0] < n_taz or od_dense.shape[1] < n_taz:
            print("      - Rellenando con ceros (padding) para coincidir con el tamaño TAZ.")
            pad_size_rows = max(0, n_taz - od_dense.shape[0])
            pad_size_cols = max(0, n_taz - od_dense.shape[1])
            od_dense = np.pad(od_dense, ((0, pad_size_rows), (0, pad_size_cols)),
                              mode='constant', constant_values=0)
        elif od_dense.shape[0] > n_taz or od_dense.shape[1] > n_taz:
            print("      - Truncando la matriz para coincidir con el tamaño TAZ.")
            od_dense = od_dense[:n_taz, :n_taz]

    # --- 6. Construcción de la Matriz OD Expandida ---

    # Dimension total de la nueva matriz: (TAZ + Aux) x (TAZ + Aux)
    n_total = n_taz + n_aux
    all_nodes = taz_nodes + aux_nodes  # El orden es TAZ primero, luego Aux

    # Inicializar la nueva matriz con el tamaño total y rellenarla completamente con NaN.
    # NaN (Not a Number) representa la demanda desconocida o no aplicable (Aux-Aux, Aux-TAZ, TAZ-Aux).
    expanded_od = np.full((n_total, n_total), np.nan, dtype=np.float32)

    # Copiar los valores de demanda conocida (TAZ-TAZ) en la esquina superior izquierda.
    # Esto sobreescribe los NaNs iniciales con los valores de demanda real.
    expanded_od[:n_taz, :n_taz] = od_dense

    # La matriz expandida está completa y lista.
    od_matrix = expanded_od  # Actualizamos la referencia al resultado final

    # --- 7. (PLACHOLDER) Actualización de DataFrame y Metadatos ---

    # PLACEHOLDER: Esta sección asume que existe un 'od_dataframe' global o de clase
    # y lo actualiza para reflejar la nueva estructura (útil para inspección en formato tabular).
    # Si 'od_dataframe' no existe, solo se muestra el mensaje de confirmación de la matriz.
    try:
        # Aquí se simularía la actualización de un DataFrame asociado
        od_dataframe = pd.DataFrame()  # Simulación de una variable global
        if not od_dataframe.empty:
            # Crear nuevo dataframe con todas las combinaciones (solo si es necesario)
            od_pairs = []
            for i, origin in enumerate(all_nodes):
                for j, dest in enumerate(all_nodes):
                    demand = expanded_od[i, j]
                    od_pairs.append({
                        'origin': origin,
                        'destination': dest,
                        'demand': demand,
                        'origin_idx': i,
                        'dest_idx': j
                    })

            od_dataframe = pd.DataFrame(od_pairs)
            print(
                f"   ✓ Matriz OD expandida y DataFrame asociado actualizado: {expanded_od.shape[0]}x{expanded_od.shape[1]}")
            print(f"      - Pares con demanda conocida: {np.sum(~np.isnan(expanded_od))}")
            print(f"      - Pares con demanda NaN: {np.sum(np.isnan(expanded_od))}")
        else:
            print(f"   ✓ Matriz OD expandida completada: {expanded_od.shape[0]}x{expanded_od.shape[1]}")
    except NameError:
        print(f"   ✓ Matriz OD expandida completada: {expanded_od.shape[0]}x{expanded_od.shape[1]}")

    return od_matrix
    # Guardar el mapeo de índices a IDs de nodos en metadata
    # TODO implementar la metadata
    # self.metadata['od_node_mapping'] = all_nodes
    # self.metadata['n_taz_nodes'] = n_taz
    # self.metadata['n_aux_nodes'] = n_aux
