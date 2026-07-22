import pandas as pd
from typing import Dict

def _network_previsualization(self, network_df: pd.DataFrame, analyzer_data: Dict):
    
    import math
    import logging
    import matplotlib.pyplot as plt
    import networkx as nx
    from pathlib import Path
    from matplotlib.lines import Line2D

    # ============================================================
    # USER-TUNABLE VISUALIZATION PARAMETERS
    # ============================================================

    # ---------- Output ----------
    output_filename = "first_network_previsualization_connectivity_control.png"
    output_dpi = 600

    # ---------- Figure ----------
    figure_width = 18
    figure_height = 14
    figure_title = "First Network Previsualization - Connectivity Control"
    figure_title_fontsize = 16
    figure_title_fontweight = "bold"

    # ---------- Colours ----------
    # Colormap used to assign unique colours to node and link types.
    # Good alternatives: "tab10", "tab20", "Set3", "Dark2".
    color_palette_name = "tab20"

    # ---------- Nodes ----------
    node_size = 450
    node_border_color = "black"
    node_border_width = 0.8

    # ---------- Node labels ----------
    show_node_labels = True
    node_label_fontsize = 8
    node_label_box_padding = 0.25
    node_label_box_facecolor = "white"
    node_label_box_edgecolor = "black"
    node_label_box_alpha = 0.85
    node_label_zorder = 5

    # ---------- Links ----------
    link_width = 2.0
    link_arrow_size = 14

    # Curvature applied to directed links.
    # Important: inverse links must use the same curvature sign to avoid overlap.
    link_curvature = 0.10

    # ---------- Link labels ----------
    show_link_labels = True
    link_label_fontsize = 7
    link_label_box_padding = 0.20
    link_label_box_facecolor = "white"
    link_label_box_alpha = 0.90
    link_label_zorder = 7

    # Multiplier controlling where the label is placed relative to the curved arc.
    # 1.0 places it approximately on the visual midpoint of the curved link.
    link_label_curve_position_factor = 1.0

    # ---------- Legend ----------
    show_legend = True
    legend_location = "upper right"
    legend_fontsize = 9
    legend_node_marker_size = 10
    legend_link_width = 3
    legend_frame = True

    # ---------- Axes ----------
    show_grid = True
    grid_linestyle = "--"
    grid_alpha = 0.25
    x_axis_label = "X coordinate"
    y_axis_label = "Y coordinate"

    # ============================================================
    # PREPARATION
    # ============================================================

    logging.info("Starting network previsualization...")

    graph = analyzer_data["graph"]

    output_dir = Path(self.scenario.config.paths.export_dirs.info_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    pos = {
        int(row["node_id"]): (float(row["x"]), float(row["y"]))
        for _, row in self.scenario.nodes_df.iterrows()
    }

    nodes_by_id = self.scenario.nodes_df.set_index("node_id")

    node_types = sorted(self.scenario.nodes_df["type"].unique())
    link_types = sorted(network_df["link_type"].unique())

    cmap = plt.get_cmap(color_palette_name)
    total_types = len(node_types) + len(link_types)

    colors = [
        cmap(i / max(total_types - 1, 1))
        for i in range(total_types)
    ]

    node_color_map = {
        node_type: colors[i]
        for i, node_type in enumerate(node_types)
    }

    link_color_map = {
        link_type: colors[len(node_types) + i]
        for i, link_type in enumerate(link_types)
    }

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # ============================================================
    # DRAW NODES
    # ============================================================

    node_colors = [
        node_color_map[nodes_by_id.loc[node, "type"]]
        for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors=node_border_color,
        linewidths=node_border_width,
        ax=ax,
    )

    # ============================================================
    # DRAW CURVED DIRECTED LINKS
    # ============================================================

    pair_curvature = {}

    for _, row in network_df.iterrows():
        u = int(row["init_node"])
        v = int(row["term_node"])
        link_type = row["link_type"]

        pair_key = tuple(sorted((u, v)))

        if pair_key not in pair_curvature:
            pair_curvature[pair_key] = link_curvature

        curvature = pair_curvature[pair_key]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v)],
            edge_color=[link_color_map[link_type]],
            width=link_width,
            arrows=True,
            arrowsize=link_arrow_size,
            connectionstyle=f"arc3,rad={curvature}",
            ax=ax,
        )

    # ============================================================
    # DRAW NODE LABELS
    # ============================================================

    if show_node_labels:
        for _, row in self.scenario.nodes_df.iterrows():
            node_id = int(row["node_id"])
            node_type = row["type"]

            x, y = pos[node_id]

            ax.text(
                x,
                y,
                f"{node_id}\n{node_type}",
                fontsize=node_label_fontsize,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle=f"round,pad={node_label_box_padding}",
                    facecolor=node_label_box_facecolor,
                    edgecolor=node_label_box_edgecolor,
                    alpha=node_label_box_alpha,
                ),
                zorder=node_label_zorder,
            )

    # ============================================================
    # DRAW LINK LABELS ON CURVED MIDPOINTS
    # ============================================================

    if show_link_labels:
        for _, row in network_df.iterrows():
            u = int(row["init_node"])
            v = int(row["term_node"])

            link_id = row["link_id"]
            link_type = row["link_type"]

            x1, y1 = pos[u]
            x2, y2 = pos[v]

            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx**2 + dy**2)

            if length == 0:
                continue

            pair_key = tuple(sorted((u, v)))
            curvature = pair_curvature[pair_key]

            # Unit perpendicular vector to the link.
            px = -dy / length
            py = dx / length

            # Straight midpoint.
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            # Approximate midpoint of the curved edge.
            curve_offset = (
                curvature
                * length
                * link_label_curve_position_factor
            )

            curved_mid_x = mx + px * curve_offset
            curved_mid_y = my + py * curve_offset

            ax.text(
                curved_mid_x,
                curved_mid_y,
                f"L{link_id}\nT{link_type}",
                fontsize=link_label_fontsize,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle=f"round,pad={link_label_box_padding}",
                    facecolor=link_label_box_facecolor,
                    edgecolor=link_color_map[link_type],
                    alpha=link_label_box_alpha,
                ),
                zorder=link_label_zorder,
            )

    # ============================================================
    # LEGEND
    # ============================================================

    if show_legend:
        node_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Node: {node_type}",
                markerfacecolor=node_color_map[node_type],
                markeredgecolor=node_border_color,
                markersize=legend_node_marker_size,
            )
            for node_type in node_types
        ]

        link_legend = [
            Line2D(
                [0],
                [0],
                color=link_color_map[link_type],
                lw=legend_link_width,
                label=f"Link: {link_type}",
            )
            for link_type in link_types
        ]

        ax.legend(
            handles=node_legend + link_legend,
            loc=legend_location,
            fontsize=legend_fontsize,
            frameon=legend_frame,
        )

    # ============================================================
    # FINAL FORMATTING
    # ============================================================

    ax.set_title(
        figure_title,
        fontsize=figure_title_fontsize,
        fontweight=figure_title_fontweight,
    )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_aspect("equal", adjustable="box")

    if show_grid:
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=output_dpi, bbox_inches="tight")
        logging.info(f"Network previsualization saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save network previsualization to {output_path}: {e}")

    plt.close(fig)

    metadata = {
        "network_visualization_path": str(output_path),
    }

    return metadata
