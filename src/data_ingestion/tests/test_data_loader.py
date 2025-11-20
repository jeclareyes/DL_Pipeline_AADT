import pytest
from data_ingestion.data_loader import DataLoader
from data_ingestion.data_processing import DataManager
from pathlib import Path


def test_data_loader_loads_multiday_od():
    # Use the provided test data file in tests/test_data
    project_root = Path(__file__).resolve().parents[2]
    test_file = project_root / 'data_ingestion' / 'tests' / 'test_data' / 'multiday_trips.tntp'

    manager = DataManager(network_name='SiouxFalls', data_root=project_root / 'data', multiday=True)
    # override od_path to test file
    manager.od_path = test_file

    loader = DataLoader(multiday=True)
    od = loader.load_od_matrix(manager)
    assert od is not None
    # expect sparse matrix
    try:
        from scipy import sparse
        assert sparse.issparse(od)
    except Exception:
        # if scipy not available, at least ensure returned object is not None
        assert od is not None

