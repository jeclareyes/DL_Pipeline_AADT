from data_ingestion.processing_modules.od_matrix_generator import MultidayODMatrixGenerator
from pathlib import Path

def test_multiday_od_generator_loads_test_file():
    project_root = Path(__file__).resolve().parents[2]
    test_file = project_root / 'data_ingestion' / 'tests' / 'test_data' / 'multiday_trips.tntp'
    gen = MultidayODMatrixGenerator(test_file)
    od_df, meta = gen.load()
    assert od_df is not None
    assert 'flow' in od_df.columns
    assert isinstance(meta, dict)

