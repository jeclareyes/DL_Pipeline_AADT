"""
Test del modelo GNN con una red simple de 3 nodos: A - B - C

Red: A -> B -> C
Demanda OD: A->C = 20
Flujo esperado: A-B = 20, B-C = 20
"""

import torch
import sys
import os

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_gnn import SimpleTrafficGNN


def create_simple_network():
    """
    Crea una red simple de 3 nodos: A - B - C

    Nodos: 0 (A), 1 (B), 2 (C)
    Links: 0->1 (A->B), 1->2 (B->C)
    """
    # Número de nodos
    num_nodes = 3

    # Features de nodos (inicialmente solo un placeholder)
    # Podríamos usar la demanda generada/atraída por cada nodo
    node_features = torch.zeros(num_nodes, 1)

    # Matriz origen-destino
    # Fila = origen, columna = destino
    od_matrix = torch.zeros(num_nodes, num_nodes)
    od_matrix[0, 2] = 20.0  # A->C = 20

    # Definir links (aristas dirigidas)
    # edge_index[0] = nodos origen, edge_index[1] = nodos destino
    edge_index = torch.tensor([
        [0, 1],  # Origen de cada link
        [1, 2]   # Destino de cada link
    ], dtype=torch.long)

    # Features de aristas (por ahora solo placeholder)
    # Podríamos incluir: capacidad, longitud, free-flow time, etc.
    num_edges = edge_index.shape[1]
    edge_features = torch.ones(num_edges, 1)

    # Flujos objetivo (ground truth)
    target_flows = torch.tensor([
        [20.0],  # A->B debe tener flujo de 20
        [20.0]   # B->C debe tener flujo de 20
    ])

    return node_features, edge_index, edge_features, od_matrix, target_flows


def print_network_info(node_features, edge_index, edge_features, od_matrix, target_flows):
    """Imprime información de la red."""
    print("=" * 60)
    print("RED DE TRANSPORTE SIMPLE: A -> B -> C")
    print("=" * 60)
    print(f"\nNúmero de nodos: {node_features.shape[0]}")
    print(f"Número de links: {edge_index.shape[1]}")

    print("\n--- LINKS ---")
    node_names = ['A', 'B', 'C']
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        print(f"Link {i}: {node_names[src]} -> {node_names[dst]}")

    print("\n--- MATRIZ ORIGEN-DESTINO ---")
    print("    ", "  ".join(node_names))
    for i, row_name in enumerate(node_names):
        row_str = f"{row_name}   " + "  ".join([f"{od_matrix[i, j].item():4.0f}" for j in range(len(node_names))])
        print(row_str)

    print("\n--- FLUJOS OBJETIVO ---")
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        print(f"Link {node_names[src]}->{node_names[dst]}: {target_flows[i, 0].item():.1f}")
    print("=" * 60)


def train_model(model, node_features, edge_index, edge_features, od_matrix, target_flows,
                num_epochs=1000, lr=0.01, alpha=10.0):
    """
    Entrena el modelo GNN.

    Args:
        alpha: Peso de la pérdida de conservación de flujo
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n🚀 INICIANDO ENTRENAMIENTO")
    print(f"Épocas: {num_epochs}, Learning Rate: {lr}, Alpha (conservación): {alpha}\n")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predicted_flows = model(node_features, edge_index, edge_features, od_matrix)

        # Calcular pérdida
        total_loss, conservation_loss, supervised_loss = model.loss(
            predicted_flows, edge_index, od_matrix, target_flows, alpha=alpha
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Imprimir progreso
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Época {epoch+1:4d} | "
                  f"Pérdida Total: {total_loss.item():8.4f} | "
                  f"Conservación: {conservation_loss.item():8.4f} | "
                  f"Supervisada: {supervised_loss.item():8.4f}")

    print("\n✅ ENTRENAMIENTO COMPLETADO\n")


def evaluate_model(model, node_features, edge_index, edge_features, od_matrix, target_flows):
    """Evalúa el modelo y muestra resultados."""
    model.eval()

    with torch.no_grad():
        predicted_flows = model(node_features, edge_index, edge_features, od_matrix)

    print("=" * 60)
    print("RESULTADOS DE LA PREDICCIÓN")
    print("=" * 60)

    node_names = ['A', 'B', 'C']
    print("\nLink          | Flujo Predicho | Flujo Objetivo | Error")
    print("-" * 60)

    total_error = 0.0
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        pred = predicted_flows[i, 0].item()
        target = target_flows[i, 0].item()
        error = abs(pred - target)
        total_error += error

        print(f"{node_names[src]}->{node_names[dst]:8s} | {pred:14.2f} | {target:14.2f} | {error:6.2f}")

    print("-" * 60)
    print(f"Error total: {total_error:.2f}")
    print(f"Error promedio: {total_error / edge_index.shape[1]:.2f}")

    # Verificar conservación de flujo
    print("\n--- VERIFICACIÓN DE CONSERVACIÓN DE FLUJO ---")
    num_nodes = node_features.shape[0]
    flow_balance = torch.zeros(num_nodes)

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        flow = predicted_flows[i, 0].item()

        flow_balance[src] -= flow  # Salida
        flow_balance[dst] += flow  # Entrada

    demand_balance = od_matrix.sum(dim=0) - od_matrix.sum(dim=1)

    print("\nNodo | Balance Flujo | Balance Demanda | Diferencia")
    print("-" * 60)
    for i, name in enumerate(node_names):
        fb = flow_balance[i].item()
        db = demand_balance[i].item()
        diff = abs(fb - db)
        print(f"{name:4s} | {fb:13.2f} | {db:15.2f} | {diff:10.2f}")

    print("=" * 60)


def main():
    """Función principal para ejecutar el test."""
    print("\n" + "🚗" * 30)
    print("TEST: MODELO GNN PARA DISTRIBUCIÓN DE FLUJO DE TRÁFICO")
    print("🚗" * 30 + "\n")

    # Crear red simple
    node_features, edge_index, edge_features, od_matrix, target_flows = create_simple_network()

    # Mostrar información de la red
    print_network_info(node_features, edge_index, edge_features, od_matrix, target_flows)

    # Crear modelo
    model = SimpleTrafficGNN(
        node_features=1,
        edge_features=1,
        hidden_dim=32
    )

    print(f"\n📊 MODELO: SimpleTrafficGNN")
    print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Entrenar modelo
    train_model(
        model, node_features, edge_index, edge_features, od_matrix, target_flows,
        num_epochs=1000,
        lr=0.01,
        alpha=10.0  # Peso alto para forzar conservación de flujo
    )

    # Evaluar modelo
    evaluate_model(model, node_features, edge_index, edge_features, od_matrix, target_flows)

    print("\n" + "🎉" * 30)
    print("TEST COMPLETADO")
    print("🎉" * 30 + "\n")


if __name__ == "__main__":
    main()

