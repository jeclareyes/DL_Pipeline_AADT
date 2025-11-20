"""
Lector de flujos de tráfico movido a processing_modules.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

# Root del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FlowReader:
    def __init__(self, path: Optional[Union[Path, str]] = None):
        if path:
            self.path = Path(path)
        else:
            self.path = PROJECT_ROOT / 'data' / 'raw' / 'Barcelona_flow.tntp'
        self.path = self.path.resolve(strict=False)
        if not self.path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.path}")
        self.flow_df = None
        self.network_df = None
        self.combined_df = None

    def load(self) -> pd.DataFrame:
        self.flow_df = pd.read_csv(self.path, delim_whitespace=True, skiprows=0)
        self.flow_df.columns = self.flow_df.columns.str.strip()
        colmap = {c.lower(): c for c in self.flow_df.columns}
        column_mapping = {}
        if 'from' in colmap:
            column_mapping[colmap['from']] = 'from_node'
        if 'to' in colmap:
            column_mapping[colmap['to']] = 'to_node'
        if 'from_node' in colmap and 'from_node' not in column_mapping:
            column_mapping[colmap['from_node']] = 'from_node'
        if 'to_node' in colmap and 'to_node' not in column_mapping:
            column_mapping[colmap['to_node']] = 'to_node'
        if 'volume' in colmap:
            column_mapping[colmap['volume']] = 'volume'
        if 'cost' in colmap:
            column_mapping[colmap['cost']] = 'cost'
        if column_mapping:
            try:
                self.flow_df.rename(columns=column_mapping, inplace=True)
            except Exception:
                pass
        if 'volume' not in self.flow_df.columns:
            vol_cols = [c for c in self.flow_df.columns if 'volume' in c.lower()]
            if vol_cols:
                def pick_volume(row):
                    for c in reversed(vol_cols):
                        v = row.get(c, None)
                        try:
                            if pd.notna(v):
                                return float(v)
                        except Exception:
                            continue
                    return 0.0
                try:
                    self.flow_df['volume'] = self.flow_df.apply(pick_volume, axis=1)
                except Exception:
                    self.flow_df['volume'] = self.flow_df[vol_cols].sum(axis=1, skipna=True)
        if 'from_node' in self.flow_df.columns:
            try:
                self.flow_df['from_node'] = pd.to_numeric(self.flow_df['from_node'], errors='coerce').astype(int)
            except Exception:
                self.flow_df['from_node'] = self.flow_df['from_node'].astype(str)
        if 'to_node' in self.flow_df.columns:
            try:
                self.flow_df['to_node'] = pd.to_numeric(self.flow_df['to_node'], errors='coerce').astype(int)
            except Exception:
                self.flow_df['to_node'] = self.flow_df['to_node'].astype(str)
        if 'volume' in self.flow_df.columns:
            try:
                self.flow_df['volume'] = pd.to_numeric(self.flow_df['volume'], errors='coerce').fillna(0.0)
            except Exception:
                pass
        if 'cost' in self.flow_df.columns:
            try:
                self.flow_df['cost'] = pd.to_numeric(self.flow_df['cost'], errors='coerce')
            except Exception:
                pass
        return self.flow_df

    def merge_with_network(self, network_path: Optional[Union[Path, str]] = None) -> pd.DataFrame:
        if self.flow_df is None:
            raise ValueError("Debe llamar a load() primero")
        if network_path is None:
            network_path = PROJECT_ROOT / 'data' / 'raw' / 'Barcelona_net.tntp'
        # lazy import to avoid circular dependency
        from .network_loader import load_network_df
        self.network_df = load_network_df(network_path)
        self.network_df.rename(columns={'init_node': 'from_node', 'term_node': 'to_node'}, inplace=True)
        self.combined_df = pd.merge(self.flow_df, self.network_df, on=['from_node', 'to_node'], how='left', indicator=True)
        self._calculate_metrics()
        return self.combined_df

    def _calculate_metrics(self):
        if self.combined_df is None:
            return
        self.combined_df['v_c_ratio'] = self.combined_df.apply(lambda r: (r['volume'] / r['capacity']) if r.get('capacity', 0) else 0.0, axis=1)
        self.combined_df['congestion_factor'] = self.combined_df.apply(lambda r: (r['cost'] / r['free_flow_time']) if r.get('free_flow_time', 0) else 1.0, axis=1)
        self.combined_df['delay'] = self.combined_df['cost'] - self.combined_df.get('free_flow_time', 0)
        self.combined_df['effective_speed'] = self.combined_df.apply(lambda r: (r.get('length', 0) / r['cost']) if r.get('cost', 0) else 0.0, axis=1)
        def classify_los(vc):
            try:
                if pd.isna(vc) or vc == 0:
                    return 'N/A'
                elif vc < 0.6:
                    return 'A'
                elif vc < 0.7:
                    return 'B'
                elif vc < 0.8:
                    return 'C'
                elif vc < 0.9:
                    return 'D'
                elif vc < 1.0:
                    return 'E'
                else:
                    return 'F'
            except Exception:
                return 'N/A'
        self.combined_df['level_of_service'] = self.combined_df['v_c_ratio'].apply(classify_los)
        self.combined_df['speed_reduction'] = self.combined_df.apply(lambda r: ((r.get('speed', 0) - r.get('effective_speed', 0)) / r.get('speed', 1)) if r.get('speed', 0) and r.get('effective_speed', 0) else 0.0, axis=1)

    def get_statistics(self) -> dict:
        if self.flow_df is None:
            raise ValueError("Debe llamar a load() primero")
        stats = {
            'total_links': len(self.flow_df),
            'links_with_flow': int((self.flow_df['volume'] > 0).sum()) if 'volume' in self.flow_df.columns else 0,
            'links_no_flow': int((self.flow_df['volume'] == 0).sum()) if 'volume' in self.flow_df.columns else 0,
            'utilization_rate': (int((self.flow_df['volume'] > 0).sum()) / len(self.flow_df) * 100) if len(self.flow_df) else 0,
            'total_volume': float(self.flow_df['volume'].sum()) if 'volume' in self.flow_df.columns else 0.0,
            'avg_volume': float(self.flow_df['volume'].mean()) if 'volume' in self.flow_df.columns else 0.0,
            'max_volume': float(self.flow_df['volume'].max()) if 'volume' in self.flow_df.columns else 0.0
        }
        if self.combined_df is not None:
            stats.update({
                'avg_v_c_ratio': float(self.combined_df['v_c_ratio'].mean()),
                'max_v_c_ratio': float(self.combined_df['v_c_ratio'].max()),
                'congested_links': int((self.combined_df['v_c_ratio'] > 0.8).sum()),
                'avg_congestion_factor': float(self.combined_df['congestion_factor'].mean()),
                'total_delay': float(self.combined_df['delay'].sum()),
            })
        return stats


def load_flow_data(path: Optional[Union[Path, str]] = None,
                   merge_network: bool = True) -> pd.DataFrame:
    reader = FlowReader(path)
    reader.load()
    if merge_network:
        return reader.merge_with_network()
    else:
        return reader.flow_df
