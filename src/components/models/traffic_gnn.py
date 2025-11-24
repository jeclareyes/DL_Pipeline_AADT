"""
Traffic Flow Distribution GNN Model

Este modelo utiliza Graph Neural Networks para distribuir flujo de tráfico
a través de los links de una red a partir de matrices origen-destino.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree


class TrafficFlowGNN(nn.Module):
    """
    GNN para distribución de flujo en redes de transporte.

    El modelo aprende a distribuir flujo desde nodos origen a nodos destino
    a través de los links de la red, respetando la conservación de flujo.
    """

    def __init__(self, node_features, edge_features, hidden_dim=64, num_layers=3):
        """
        Args:
            node_features: Dimensión de features de nodos (ej: demanda OD)
            edge_features: Dimensión de features de aristas (ej: capacidad, longitud)
            hidden_dim: Dimensión de la capa oculta
            num_layers: Número de capas de message passing
        """
        super(TrafficFlowGNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Encoder para nodos - incluye información de OD
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Encoder para matriz OD (información adicional)
        self.od_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # demanda generada + atraída
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Encoder para aristas
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Capas GNN
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Decoder para predecir flujo en cada arista
        self.flow_decoder = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),  # source + target + edge
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Flujo no puede ser negativo
        )

    def forward(self, x, edge_index, edge_attr, od_matrix=None):
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            od_matrix: Matriz OD [num_nodes, num_nodes] (opcional)

        Returns:
            edge_flows: Flujo predicho para cada arista [num_edges, 1]
        """
        num_nodes = x.shape[0]

        # Codificar features básicas de nodos
        node_emb = self.node_encoder(x)

        # Si tenemos matriz OD, agregar esa información
        if od_matrix is not None:
            # Demanda generada (filas) y atraída (columnas) por cada nodo
            demand_generated = od_matrix.sum(dim=1).unsqueeze(1)  # [num_nodes, 1]
            demand_attracted = od_matrix.sum(dim=0).unsqueeze(1)  # [num_nodes, 1]
            od_features = torch.cat([demand_generated, demand_attracted], dim=1)

            od_emb = self.od_encoder(od_features)
            node_emb = node_emb + od_emb  # Combinar información

        # Codificar aristas
        edge_emb = self.edge_encoder(edge_attr)

        # Message passing
        for conv in self.convs:
            node_emb = F.relu(conv(node_emb, edge_index))

        # Predecir flujo en cada arista
        edge_flows = self._decode_flows(node_emb, edge_index, edge_emb)

        return edge_flows

    def _decode_flows(self, node_emb, edge_index, edge_emb):
        """
        Decodifica flujos en aristas a partir de embeddings.
        """
        # Obtener embeddings de nodos origen y destino para cada arista
        src_emb = node_emb[edge_index[0]]  # Nodos origen
        dst_emb = node_emb[edge_index[1]]  # Nodos destino

        # Concatenar y decodificar
        flow_input = torch.cat([src_emb, dst_emb, edge_emb], dim=-1)
        edge_flows = self.flow_decoder(flow_input)

        return edge_flows

    def compute_flow_conservation_loss(self, edge_flows, edge_index, od_matrix):
        """
        Calcula la pérdida de conservación de flujo.

        Para cada nodo, el flujo entrante - flujo saliente debe ser igual
        a la suma de demandas que terminan en ese nodo menos las que comienzan.
        """
        num_nodes = od_matrix.shape[0]
        device = edge_flows.device

        # Calcular flujo neto en cada nodo
        flow_balance = torch.zeros(num_nodes, device=device)

        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            flow = edge_flows[i, 0]

            # Flujo saliente del nodo origen
            flow_balance[src] -= flow
            # Flujo entrante al nodo destino
            flow_balance[dst] += flow

        # Calcular demanda neta en cada nodo (destinos - orígenes)
        demand_balance = od_matrix.sum(dim=0) - od_matrix.sum(dim=1)

        # La conservación de flujo requiere que flow_balance = demand_balance
        conservation_loss = F.mse_loss(flow_balance, demand_balance)

        return conservation_loss


class SimpleTrafficGNN(nn.Module):
    """
    Versión simplificada del modelo para casos básicos.
    """

    def __init__(self, node_features=1, edge_features=1, hidden_dim=32):
        super(SimpleTrafficGNN, self).__init__()

        self.gnn = TrafficFlowGNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=2
        )

    def forward(self, x, edge_index, edge_attr, od_matrix):
        return self.gnn(x, edge_index, edge_attr, od_matrix)

    def loss(self, edge_flows, edge_index, od_matrix, target_flows=None, alpha=1.0):
        """
        Calcula pérdida total.

        Args:
            edge_flows: Flujos predichos
            edge_index: Conectividad
            od_matrix: Matriz OD
            target_flows: Flujos objetivo (si disponibles)
            alpha: Peso de la pérdida de conservación
        """
        # Pérdida de conservación de flujo
        conservation_loss = self.gnn.compute_flow_conservation_loss(
            edge_flows, edge_index, od_matrix
        )

        total_loss = alpha * conservation_loss

        # Si tenemos flujos objetivo, agregar pérdida supervisada
        if target_flows is not None:
            supervised_loss = F.mse_loss(edge_flows, target_flows)
            total_loss = total_loss + supervised_loss
            return total_loss, conservation_loss, supervised_loss

        return total_loss, conservation_loss, torch.tensor(0.0)
