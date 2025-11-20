"""
Generador de matriz Origen-Destino (OD) desde archivos TNTP.

Copiado desde src/data_ingestion/od_matrix_generator.py para alojarlo en processing_modules.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')

# Root del proyecto (dos niveles arriba de src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class MultidayODMatrixGenerator:
    def __init__(self, path: Optional[Union[Path, str]] = None):
        if path:
            self.path = Path(path)
        else:
            project_root = Path(__file__).resolve().parents[2]
            self.path = project_root / 'data' / 'raw' / 'trips.tntp'

        self.path = self.path.resolve(strict=False)
        if not self.path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.path}")

        self.metadata = {}
        self.od_dataframe = None
        self.daily_matrices = {}
        self.hourly_data = []
        self.od_matrix_sparse = None

    def _read_lines(self) -> List[str]:
        with open(self.path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()

    def _parse_metadata(self, lines: List[str]) -> Dict[str, Union[int, float, List[str]]]:
        metadata = {}
        import re
        metadata_dates = []
        for line in lines:
            line = line.strip()
            if line.startswith('<NUMBER OF ZONES>'):
                try:
                    metadata['zones'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif line.lower().startswith('<total od flow'):
                # handle variants like '<TOTAL OD FLOW 2022-09-26> 406722.0'
                parts = line.split()
                try:
                    # last token should be the numeric total flow
                    metadata['total_flow'] = float(parts[-1])
                except Exception:
                    pass
                # try to capture a date present inside the tag
                found = re.findall(r"\d{4}-\d{2}-\d{2}", line)
                for d in found:
                    if d not in metadata_dates:
                        metadata_dates.append(d)
            elif line == '<END OF METADATA>':
                break
        if metadata_dates:
            metadata['dates_in_metadata'] = metadata_dates
        return metadata

    def _parse_datetime(self, line: str) -> Optional[Tuple[str, str, Optional[int]]]:
        import re
        # Accept variants like '<matrix> YYYY-MM-DD HH:MM Origin N',
        # '<matrix date> YYYY-MM-DD HH:MM' or '<MATRIX DATE> 2022-09-26 00:00'
        # Allow optional closing '>' before the date and optional Origin on same line
        pattern = r"<\s*matrix(?:\s+date)?\s*>?\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})(?:\s+Origin\s+(\d+))?"
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            origin_group = match.group(3)
            origin = int(origin_group) if origin_group is not None else None
            datetime_str = f"{date_str} {time_str}"
            return date_str, datetime_str, origin
        return None

    def _parse_multiday_od_data(self, lines: List[str]) -> List[Dict]:
        """Parse the file by matrix blocks. Each block starts at a '<matrix' header and continues
        until the next '<matrix' header or EOF. This approach is robust to different header
        formats where 'Origin' may be on the next line.
        """
        import re
        od_data = []
        data_started = False

        # find index of end of metadata
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == '<END OF METADATA>':
                start_idx = i + 1
                break

        i = start_idx
        n = len(lines)
        header_re = re.compile(r"<\s*matrix(?:\s+date)?\s*>?\s*(\d{4}-\d{2}-\d{2})?\s*(\d{2}:\d{2})?", re.IGNORECASE)

        while i < n:
            line = lines[i].strip()
            # find next header
            if line.lower().startswith('<matrix'):
                # extract date/time if present
                m = header_re.match(line)
                if m:
                    date_str = m.group(1)
                    time_str = m.group(2)
                else:
                    date_str = None
                    time_str = None

                # collect block lines until next '<matrix' header or EOF
                block_lines = []
                i += 1
                while i < n and not lines[i].strip().lower().startswith('<matrix'):
                    block_lines.append(lines[i].rstrip('\n'))
                    i += 1

                # parse block: lines may contain 'Origin N' lines followed by OD pairs lines
                current_origin = None
                for bl in block_lines:
                    s = bl.strip()
                    if not s:
                        continue
                    if s.lower().startswith('origin'):
                        parts = s.split()
                        if len(parts) >= 2:
                            try:
                                current_origin = int(parts[1])
                            except Exception:
                                current_origin = None
                        continue
                    if current_origin is not None and ':' in s:
                        pairs = s.split(';')
                        for pair in pairs:
                            pair = pair.strip()
                            if ':' in pair:
                                try:
                                    dest_str, flow_str = pair.split(':', 1)
                                    destination = int(dest_str.strip())
                                    flow = float(flow_str.strip())
                                    if flow > 0:
                                        od_data.append({
                                            'date': date_str,
                                            'datetime': f"{date_str} {time_str}" if date_str and time_str else None,
                                            'origin': current_origin,
                                            'destination': destination,
                                            'flow': flow
                                        })
                                except Exception:
                                    continue
                # continue loop without increment (already at next header or EOF)
                continue
            else:
                i += 1
        return od_data

    def load(self) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, List[str]]]]:
        print("\n📅 Procesando matriz OD multiday...")
        lines = self._read_lines()
        self.metadata = self._parse_metadata(lines)
        self.hourly_data = self._parse_multiday_od_data(lines)
        if not self.hourly_data:
            raise ValueError("No se encontraron datos OD en el archivo")
        hourly_df = pd.DataFrame(self.hourly_data)
        # If parsing failed to capture dates/timestamps but metadata contains a date,
        # use it as a fallback so we can aggregate per-day. Some TNTP variants place
        # the date in metadata tags rather than per-matrix headers.
        if 'dates_in_metadata' in self.metadata and hourly_df['date'].notna().sum() == 0:
            fallback_date = self.metadata['dates_in_metadata'][0]
            hourly_df['date'] = fallback_date
            # set a simple datetime if missing
            if 'datetime' not in hourly_df or hourly_df['datetime'].isna().all():
                hourly_df['datetime'] = hourly_df['date'].astype(str) + ' 00:00'
        print(f"   ✓ Datos horarios parseados: {len(hourly_df)} entradas")
        print(f"   ✓ Fechas únicas: {hourly_df['date'].nunique()}")
        print(f"   ✓ Timestamps únicos: {hourly_df['datetime'].nunique()}")
        print("\n   📊 Agregando matrices por día...")
        daily_dfs = []
        for date in hourly_df['date'].unique():
            date_data = hourly_df[hourly_df['date'] == date]
            daily_matrix = date_data.groupby(['origin', 'destination'], as_index=False)['flow'].sum()
            daily_matrix['date'] = date
            daily_dfs.append(daily_matrix)
            self.daily_matrices[date] = daily_matrix
            print(f"      - {date}: {len(daily_matrix)} pares OD, flujo total: {daily_matrix['flow'].sum():.2f}")
        print("\n   📈 Calculando matriz promedio diario...")
        all_daily = pd.concat(daily_dfs, ignore_index=True)
        n_days = len(self.daily_matrices)
        avg_matrix = all_daily.groupby(['origin', 'destination'], as_index=False)['flow'].sum()
        avg_matrix['flow'] = avg_matrix['flow'] / n_days
        self.od_dataframe = avg_matrix[['origin', 'destination', 'flow']]
        self.metadata['n_days'] = n_days
        self.metadata['n_hourly_records'] = len(hourly_df)
        self.metadata['n_daily_od_pairs'] = len(avg_matrix)
        self.metadata['avg_daily_flow'] = avg_matrix['flow'].sum()
        self.metadata['dates'] = sorted(self.daily_matrices.keys())
        print(f"\n   ✓ Matriz promedio diario generada:")
        print(f"      - Días procesados: {n_days}")
        print(f"      - Pares OD únicos: {len(avg_matrix)}")
        print(f"      - Flujo diario promedio: {self.metadata['avg_daily_flow']:.2f}")
        return self.od_dataframe, self.metadata

    def to_sparse_matrix(self, format: str = 'csr') -> sparse.spmatrix:
        if self.od_dataframe is None:
            raise ValueError("Debe llamar a load() primero")
        n_zones = self.metadata.get('zones', self.od_dataframe[['origin', 'destination']].max().max())
        origins = self.od_dataframe['origin'].values - 1
        destinations = self.od_dataframe['destination'].values - 1
        flows = self.od_dataframe['flow'].values
        matrix_coo = sparse.coo_matrix((flows, (origins, destinations)), shape=(n_zones, n_zones), dtype=np.float32)
        if format == 'csr':
            matrix = matrix_coo.tocsr()
        elif format == 'csc':
            matrix = matrix_coo.tocsc()
        elif format == 'coo':
            matrix = matrix_coo
        elif format == 'lil':
            matrix = matrix_coo.tolil()
        else:
            raise ValueError(f"Formato no soportado: {format}")
        self.od_matrix_sparse = matrix
        return matrix

    def save_sparse(self, output_path: Union[Path, str], compressed: bool = True):
        if self.od_matrix_sparse is None:
            self.to_sparse_matrix()
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            sparse.save_npz(output_path, self.od_matrix_sparse, compressed=True)
        else:
            sparse.save_npz(output_path, self.od_matrix_sparse, compressed=False)
        metadata_path = output_path.with_suffix('.meta.npz')
        np.savez(metadata_path,
                 zones=self.metadata.get('zones', 0),
                 n_days=self.metadata.get('n_days', 0),
                 avg_daily_flow=self.metadata.get('avg_daily_flow', 0.0),
                 n_daily_od_pairs=self.metadata.get('n_daily_od_pairs', 0),
                 n_hourly_records=self.metadata.get('n_hourly_records', 0),
                 sparsity=(1.0 - self.od_matrix_sparse.nnz / (self.od_matrix_sparse.shape[0] * self.od_matrix_sparse.shape[1])))

    def save_dataframe(self, output_path: Union[Path, str], format: str = 'parquet'):
        if self.od_dataframe is None:
            raise ValueError("Debe llamar a load() primero")
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format == 'parquet':
            self.od_dataframe.to_parquet(output_path, compression='gzip', index=False)
        elif format == 'csv':
            self.od_dataframe.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Formato no soportado: {format}")

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        if self.od_dataframe is None:
            raise ValueError("Debe llamar a load() primero")
        stats = {
            'n_days': self.metadata.get('n_days', 0),
            'n_hourly_records': self.metadata.get('n_hourly_records', 0),
            'n_daily_od_pairs': len(self.od_dataframe),
            'avg_daily_flow': self.metadata.get('avg_daily_flow', 0.0),
            'mean_flow': self.od_dataframe['flow'].mean(),
            'median_flow': self.od_dataframe['flow'].median(),
            'min_flow': self.od_dataframe['flow'].min(),
            'max_flow': self.od_dataframe['flow'].max(),
            'std_flow': self.od_dataframe['flow'].std(),
        }
        n_zones = stats.get('n_zones', self.metadata.get('zones', 0))
        possible_pairs = n_zones * n_zones if n_zones else 0
        stats['sparsity'] = 1.0 - (stats['n_daily_od_pairs'] / possible_pairs) if possible_pairs else 1.0
        stats['density'] = stats['n_daily_od_pairs'] / possible_pairs if possible_pairs else 0.0
        return stats

class ODMatrixGenerator:
    def __init__(self, path: Optional[Union[Path, str]] = None):
        if path:
            self.path = Path(path)
        else:
            project_root = Path(__file__).resolve().parents[2]
            self.path = project_root / 'data' / 'raw' / 'Barcelona_trips.tntp'
        self.path = self.path.resolve(strict=False)
        if not self.path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.path}")
        self.metadata = {}
        self.od_dataframe = None
        self.od_matrix_dense = None
        self.od_matrix_sparse = None

    def _read_lines(self) -> List[str]:
        with open(self.path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()

    def _parse_metadata(self, lines: List[str]) -> Dict[str, Union[int, float]]:
        metadata = {}
        for line in lines:
            line = line.strip()
            if line.startswith('<NUMBER OF ZONES>'):
                try:
                    metadata['zones'] = int(line.split()[-1])
                except Exception:
                    pass
            elif line.startswith('<TOTAL OD FLOW>'):
                try:
                    metadata['total_flow'] = float(line.split()[-1])
                except Exception:
                    pass
            elif line == '<END OF METADATA>':
                break
        return metadata

    def _parse_od_data(self, lines: List[str]) -> pd.DataFrame:
        od_data = []
        current_origin = None
        data_start = False
        for line in lines:
            line = line.strip()
            if line == '<END OF METADATA>':
                data_start = True
                continue
            if not data_start or not line:
                continue
            if line.startswith('Origin'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current_origin = int(parts[1])
                    except ValueError:
                        pass
                continue
            if current_origin is not None and ':' in line:
                pairs = line.split(';')
                for pair in pairs:
                    pair = pair.strip()
                    if ':' in pair:
                        try:
                            dest_str, flow_str = pair.split(':')
                            destination = int(dest_str.strip())
                            flow = float(flow_str.strip())
                            if flow > 0:
                                od_data.append({'origin': current_origin, 'destination': destination, 'flow': flow})
                        except Exception:
                            continue
        return pd.DataFrame(od_data)

    def load(self) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
        lines = self._read_lines()
        self.metadata = self._parse_metadata(lines)
        self.od_dataframe = self._parse_od_data(lines)
        return self.od_dataframe, self.metadata

    def to_dense_matrix(self, validate: bool = True) -> np.ndarray:
        if self.od_dataframe is None:
            raise ValueError("Debe llamar a load() primero")
        n_zones = self.metadata.get('zones', self.od_dataframe[['origin', 'destination']].max().max())
        matrix = np.zeros((n_zones, n_zones), dtype=np.float32)
        for _, row in self.od_dataframe.iterrows():
            origin_idx = int(row['origin']) - 1
            dest_idx = int(row['destination']) - 1
            matrix[origin_idx, dest_idx] = row['flow']
        if validate and 'total_flow' in self.metadata:
            calculated_total = matrix.sum()
            metadata_total = self.metadata['total_flow']
            error = abs(calculated_total - metadata_total)
            if error > 0.01:
                warnings.warn(f"Error de balance: calculado={calculated_total:.2f}, esperado={metadata_total:.2f}, diferencia={error:.6f}")
        self.od_matrix_dense = matrix
        return matrix

    def to_sparse_matrix(self, format: str = 'csr') -> sparse.spmatrix:
        if self.od_dataframe is None:
            raise ValueError("Debe llamar a load() primero")
        n_zones = self.metadata.get('zones', self.od_dataframe[['origin', 'destination']].max().max())
        origins = self.od_dataframe['origin'].values - 1
        destinations = self.od_dataframe['destination'].values - 1
        flows = self.od_dataframe['flow'].values
        matrix_coo = sparse.coo_matrix((flows, (origins, destinations)), shape=(n_zones, n_zones), dtype=np.float32)
        if format == 'csr':
            matrix = matrix_coo.tocsr()
        elif format == 'csc':
            matrix = matrix_coo.tocsc()
        elif format == 'coo':
            matrix = matrix_coo
        elif format == 'lil':
            matrix = matrix_coo.tolil()
        else:
            raise ValueError(f"Formato no soportado: {format}. Use 'csr', 'csc', 'coo', o 'lil'")
        self.od_matrix_sparse = matrix
        return matrix

    def save_sparse(self, output_path: Union[Path, str], compressed: bool = True):
        if self.od_matrix_sparse is None:
            self.to_sparse_matrix()
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            sparse.save_npz(output_path, self.od_matrix_sparse, compressed=True)
        else:
            sparse.save_npz(output_path, self.od_matrix_sparse, compressed=False)
        metadata_path = output_path.with_suffix('.meta.npz')
        np.savez(metadata_path, zones=self.metadata.get('zones', 0), total_flow=self.metadata.get('total_flow', 0.0), n_od_pairs=len(self.od_dataframe), sparsity=(1.0 - self.od_matrix_sparse.nnz / (self.od_matrix_sparse.shape[0] * self.od_matrix_sparse.shape[1])))

    def save_dataframe(self, output_path: Union[Path, str], format: str = 'parquet'):
        if self.od_dataframe is None:
            raise ValueError("Debe llamar a load() primero")
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format == 'parquet':
            self.od_dataframe.to_parquet(output_path, compression='gzip', index=False)
        elif format == 'csv':
            self.od_dataframe.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Formato no soportado: {format}. Use 'parquet' o 'csv'")

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        if self.od_dataframe is None:
            raise ValueError("Debe llamar a load() primero")
        stats = {
            'n_zones': self.metadata.get('zones', 0),
            'n_od_pairs': len(self.od_dataframe),
            'total_flow': self.od_dataframe['flow'].sum(),
            'metadata_total_flow': self.metadata.get('total_flow', 0.0),
            'mean_flow': self.od_dataframe['flow'].mean(),
            'median_flow': self.od_dataframe['flow'].median(),
            'min_flow': self.od_dataframe['flow'].min(),
            'max_flow': self.od_dataframe['flow'].max(),
            'std_flow': self.od_dataframe['flow'].std(),
        }
        n_zones = stats['n_zones']
        possible_pairs = n_zones * n_zones
        stats['sparsity'] = 1.0 - (stats['n_od_pairs'] / possible_pairs)
        stats['density'] = stats['n_od_pairs'] / possible_pairs
        return stats


def load_od_matrix(path: Optional[Union[Path, str]] = None,
                   format: str = 'sparse') -> Union[np.ndarray, sparse.spmatrix, pd.DataFrame]:
    generator = ODMatrixGenerator(path)
    generator.load()
    if format == 'sparse':
        return generator.to_sparse_matrix()
    elif format == 'dense':
        return generator.to_dense_matrix()
    elif format == 'dataframe':
        return generator.od_dataframe
    else:
        raise ValueError(f"Formato no soportado: {format}. Use 'sparse', 'dense', o 'dataframe'")
