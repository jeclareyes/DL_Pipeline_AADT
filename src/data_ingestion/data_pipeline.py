"""
Simple data pipeline runner for full end-to-end processing.
Saves outputs into data/processed/<case> using DataSaver.
"""
from pathlib import Path
import sys
import argparse

# Ensure package imports work when run as script
src_dir = str(Path(__file__).resolve().parents[1])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_ingestion.data_processing import DataManager, _read_yaml_defaults
from data_ingestion.data_loader import DataLoader
from data_ingestion.data_saver import DataSaver

# Read YAML defaults from configs/linkoping.yaml if present
script_dir = Path(__file__).resolve().parent
yaml_file = script_dir.parents[1] / 'configs' / 'linkoping.yaml'
yaml_defaults = _read_yaml_defaults(yaml_file) if yaml_file.exists() else {}


def run_pipeline(case: str, source: str, multiday: bool, save_graph_pickle: bool, save_graph_image: bool):
    # This function does not provide default parameter values; all defaults
    # must be provided by the caller (typically resolved from YAML). The
    # pipeline always prefers values from `configs/linkoping.yaml` when the
    # caller passes None for an argument.
    # Resolve parameters against YAML defaults here to ensure a single place
    # decides final values.
    # If the caller passed explicit non-None values, those values are used.
    routing_config = yaml_defaults.get('routing', {})
    case = case if case is not None else routing_config.get('case')
    source = source if source is not None else routing_config.get('route_source')
    multiday = multiday if multiday is not None else bool(routing_config.get('multiday', False))
    save_graph_pickle = save_graph_pickle if save_graph_pickle is not None else routing_config.get('save_graph')
    save_graph_image = save_graph_image if save_graph_image is not None else routing_config.get('save_image')

    # After resolving, case must be defined
    if case is None:
        raise ValueError('A `case` must be provided either via arguments or configs/linkoping.yaml')

    print(f"Running pipeline for case={case} source={source} multiday={multiday}")
    data_root = None
    if source is not None:
        data_root = Path(__file__).resolve().parents[2] / 'data' / source

    manager = DataManager(network_name=case, data_root=data_root, multiday=multiday)

    # Use canonical name for output filenames
    case_name = manager.canonical_name if manager.canonical_name else case

    loader = DataLoader(multiday=multiday)
    print("--- Loading data ---")
    loader.load_all(manager)

    print("--- Merging network and flows ---")
    unified = manager.merge_network_flow()

    print("--- Building graph ---")
    graph = manager.build_graph()

    print("--- Processing routes (k-shortest paths) ---")
    # Always take routing configuration from the YAML; do not hardcode defaults here.
    routing_config = yaml_defaults.get('routing', {})
    route_source = routing_config.get('route_source')
    num_routes = routing_config.get('num_routes')
    route_algorithm = routing_config.get('route_algorithm')
    execution_mode = routing_config.get('execution_mode')
    use_route_cache = routing_config.get('use_route_cache')
    # Support both 'force_recompute' and legacy/mistyped 'force_precompute' keys
    force_recompute = routing_config.get('force_recompute') if 'force_recompute' in routing_config else routing_config.get('force_precompute')
    routes_data = manager.process_routes(
        graph,
        route_source=route_source,
        num_routes=num_routes,
        route_algorithm=route_algorithm,
        execution_mode=execution_mode,
        weight='free_flow_time',
        force_recompute=force_recompute,
        use_route_cache=use_route_cache,
    )

    print("--- Saving outputs ---")
    saver = DataSaver(manager.processed_dir)

    # Save graph with case-prefixed name
    if save_graph_pickle:
        graph_path = manager.processed_dir / f'{case_name}_graph.pkl'
        p = saver.save_graph_pickle(graph, graph_path)
        print(f"Saved graph pickle: {p}")
    if save_graph_image:
        image_path = manager.processed_dir / f'{case_name}_graph.png'
        p2 = saver.save_graph_image(graph, image_path, fmt='png')
        print(f"Saved graph image: {p2}")

    # Save unified dataframe as parquet with case-prefixed name
    try:
        out_df_path = manager.processed_dir / f'{case_name}_link_data.parquet'
        unified.to_parquet(out_df_path, index=False)
        print(f"Saved link data: {out_df_path}")
    except Exception as e:
        print(f"Could not save link data: {e}")

    # Save OD matrix (sparse format)
    try:
        if manager.od_matrix is not None:
            od_path = manager.processed_dir / f'{case_name}_od_matrix.npz'
            saver.save_od_matrix(manager.od_matrix, od_path, fmt='sparse')
            print(f"Saved OD matrix: {od_path}")

        # Also save OD dataframe if available
        if manager.od_dataframe is not None:
            od_df_path = manager.processed_dir / f'{case_name}_od_dataframe.parquet'
            manager.od_dataframe.to_parquet(od_df_path, index=False)
            print(f"Saved OD dataframe: {od_df_path}")
    except Exception as e:
        print(f"Could not save OD matrix: {e}")

    print("Pipeline completed.")
    return manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data pipeline and save outputs to data/processed/<case>')
    # Do not hardcode defaults in the CLI; use YAML values when args are omitted.
    parser.add_argument('case', nargs='?', default=None)
    parser.add_argument('--source', choices=['external', 'interim', 'raw'], default=None)
    parser.add_argument('--multiday', action='store_true')
    parser.add_argument('--no-pickle', dest='pickle', action='store_false', help='Do not save graph pickle', default=None)
    parser.add_argument('--image', action='store_true', help='Save graph image', default=None)
    args = parser.parse_args()

    # Resolve CLI args against YAML defaults (pass None for values we want the
    # function to pick from YAML when the user didn't provide them explicitly).
    case_arg = args.case if args.case is not None else None
    source_arg = args.source if args.source is not None else None
    multiday_arg = args.multiday if args.multiday is not None else None
    # For boolean flags, argparse sets False when not provided for store_true/store_false.
    # We want to allow YAML to control defaults, so interpret missing flags as None.
    # `--multiday` uses store_true, so if user did not pass it, args.multiday is False.
    # Detect presence by inspecting sys.argv.
    import sys as _sys
    multiday_present = '--multiday' in _sys.argv
    multiday_arg = args.multiday if multiday_present else None

    pickle_present = '--no-pickle' in _sys.argv or '--pickle' in _sys.argv
    # args.pickle default None; if user passed --no-pickle it will be False, else None
    save_pickle_arg = args.pickle if pickle_present else None

    image_present = '--image' in _sys.argv
    save_image_arg = args.image if image_present else None

    run_pipeline(case=case_arg, source=source_arg, multiday=multiday_arg, save_graph_pickle=save_pickle_arg, save_graph_image=save_image_arg)
