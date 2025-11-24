

# 2) Grafo with numeric node ids (ints)
G2 = nx.DiGraph()
G2.add_edge(1, 2, free_flow_time=1.0)
G2.add_edge(2, 2, free_flow_time=1.0)
run_test(G2, 'Node IDs as ints; includes self-loops')

# 3) Graph where node types missing so _get_od_pairs fallback uses first 50 nodes
G3 = nx.DiGraph()
G3.add_edge('a', 'b', free_flow_time=1.0)
G3.add_edge('b', 'c', free_flow_time=1.0)
# Do not set node type to force fallback condition
handler3 = RouteHandler(graph=G3, node_df=None, output_route='outputs/test_routes.pkl', config=config)
print('\n--- Test: fallback when no taz/aux present (should warn)')
print('OD pairs length:', len(handler3._get_od_pairs()))

print('\nDone tests')
