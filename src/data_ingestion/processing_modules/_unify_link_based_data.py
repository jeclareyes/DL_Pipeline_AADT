def unify_link_data(network_df, flow_df):
    """Combina los datos de la red y los flujos en un único DataFrame unificado.

    This function is robust to network DataFrame using 'init_node'/'term_node'
    column names (common in TNTP loaders) or already using 'from_node'/'to_node'.
    It performs a LEFT merge keeping all flow rows.
    """
    import pandas as pd

    if network_df is None or flow_df is None:
        raise RuntimeError("Debe cargar la red y los flujos antes de fusionarlos.")

    # Work on a copy of network to avoid mutating original
    network_renamed = network_df.copy()

    # If network uses TNTP standard names, rename to from_node/to_node
    if 'init_node' in network_renamed.columns and 'term_node' in network_renamed.columns:
        network_renamed.rename(columns={'init_node': 'from_node', 'term_node': 'to_node'}, inplace=True)

    # Drop any duplicated column labels to avoid merge errors (do after rename)
    if network_renamed.columns.duplicated().any():
        network_renamed = network_renamed.loc[:, ~network_renamed.columns.duplicated()]

    # Ensure both frames have the join columns
    if 'from_node' not in network_renamed.columns or 'to_node' not in network_renamed.columns:
        raise RuntimeError("La tabla de red no contiene columnas 'from_node'/'to_node' ni 'init_node'/'term_node'.")
    if 'from_node' not in flow_df.columns or 'to_node' not in flow_df.columns:
        raise RuntimeError("La tabla de flujos no contiene columnas 'from_node'/'to_node'.")

    # Ensure flow_df has no duplicated columns either
    if flow_df.columns.duplicated().any():
        flow_df = flow_df.loc[:, ~flow_df.columns.duplicated()]

    # Normalize types to string for a safe merge
    def _to_str_col(df, col):
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col]).astype(int).astype(str)
        except Exception:
            df[col] = df[col].astype(str)

    _to_str_col(network_renamed, 'from_node')
    _to_str_col(network_renamed, 'to_node')
    _to_str_col(flow_df, 'from_node')
    _to_str_col(flow_df, 'to_node')

    # Perform LEFT merge to keep all flows
    merged_df = pd.merge(
        flow_df,
        network_renamed,
        on=['from_node', 'to_node'],
        how='left',
        indicator=True
    )

    # Verificar si hay columnas duplicadas después de la fusión
    duplicated_columns = merged_df.columns[merged_df.columns.duplicated()].tolist()
    if duplicated_columns:
        print(f"Advertencia: Se encontraron columnas duplicadas en la fusión: {duplicated_columns}")

    # Normalize lanes and VDF if present
    if 'lanes' in merged_df.columns:
        try:
            merged_df['lanes'] = pd.to_numeric(merged_df['lanes'], errors='coerce').fillna(1).astype(int)
        except Exception:
            merged_df['lanes'] = merged_df['lanes'].astype(int, errors='ignore')

    if 'VDF' in merged_df.columns:
        merged_df['VDF'] = pd.to_numeric(merged_df['VDF'], errors='coerce')

    # Reorder columns: place 'lanes' after 'capacity' and 'VDF' after 'speed' when present
    cols = list(merged_df.columns)

    def move_after(lst, col_to_move, after_col):
        if col_to_move in lst and after_col in lst:
            lst.remove(col_to_move)
            idx = lst.index(after_col)
            lst.insert(idx + 1, col_to_move)
        return lst

    cols = move_after(cols, 'lanes', 'capacity')
    cols = move_after(cols, 'VDF', 'speed')
    merged_df = merged_df[cols]

    # Actualizar los metadatos para reflejar la fusión
    # TODO implementar lo de metadata
    # metadata['merged_columns'] = list(merged_df.columns)

    # Store unified_df
    unified_df = _deduplicate_link_data(merged_df)

    return unified_df


def _deduplicate_link_data(link_df):
    """Elimina filas duplicadas en el DataFrame de enlaces basado en from_node y to_node.

    Regla: para cada par (from_node, to_node) si existe al menos una lectura con
    `link_type != 99` se conservará una de esas lecturas (y se eliminarán las de
    `link_type == 99`). Si todas las lecturas del par tienen `link_type == 99`,
    se conservará la primera ocurrencia.

    Al final imprime cuántas lecturas se removieron, cuántos registros quedan y
    un mini-DataFrame con las lecturas eliminadas (primeras 10 filas).
    """
    import pandas as pd

    if link_df is None:
        raise RuntimeError("Debe proporcionar un DataFrame de enlaces para deduplicar.")

    df = link_df.copy()

    # Ensure join columns exist
    if 'from_node' not in df.columns or 'to_node' not in df.columns:
        raise RuntimeError("El DataFrame debe contener las columnas 'from_node' y 'to_node'.")

    # We'll collect indices to keep and to remove
    to_keep = []
    to_remove = []

    # Preserve original order by grouping using the existing index order
    grouped = df.groupby(['from_node', 'to_node'], sort=False, dropna=False)

    for _, group in grouped:
        # If link_type column exists, prefer rows where link_type != 99
        if 'link_type' in group.columns:
            non99 = group[group['link_type'] != 99]
            if len(non99) > 0:
                # Keep the first non-99 occurrence (preserving original order)
                keep_idx = non99.index[0]
            else:
                # All are 99 -> keep the first row in the original group
                keep_idx = group.index[0]
        else:
            # No link_type info -> keep first row
            keep_idx = group.index[0]

        # Mark indices
        to_keep.append(keep_idx)
        # All other rows in the group are removed
        removes = [i for i in group.index if i != keep_idx]
        to_remove.extend(removes)

    # Build deduplicated and removed DataFrames
    deduplicated_df = df.loc[to_keep].reset_index(drop=True)
    removed_df = df.loc[to_remove].reset_index(drop=True) if to_remove else pd.DataFrame(columns=df.columns)

    removed_count = len(removed_df)
    remaining_count = len(deduplicated_df)

    print(f"Deduplicación: se removieron {removed_count} filas; quedan {remaining_count} registros en el dataset.")

    """ 
    if removed_count > 0:
        # Show a small sample of removed rows (up to 10)
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
            print("Muestra de lecturas eliminadas (hasta 10 filas):")
            print(removed_df.head(10).to_string(index=False))
    """
    return deduplicated_df
