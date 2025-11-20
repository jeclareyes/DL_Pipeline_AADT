def test_basic_imports():
    # Sanity check: import key modules via package
    import importlib
    import data_ingestion
    mod_names = [
        'data_ingestion.data_processing',
        'data_ingestion.data_loader',
        'data_ingestion.data_saver',
        'data_ingestion.processing_modules.network_loader',
        'data_ingestion.processing_modules.od_matrix_generator'
    ]
    for m in mod_names:
        mod = importlib.import_module(m)
        assert mod is not None

