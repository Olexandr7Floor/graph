import graphviz
import csv
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import sys
def read_graph_data(file_path):
    data = set()
    unique_nodes = set()
    node_mentions = {}
    repeated_edges = {}
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(row) != 2:
                continue
            from_node, to_node = [node.strip().lower() for node in row]

            if from_node not in unique_nodes:
                unique_nodes.add(from_node)
                node_mentions[from_node] = 1
            else:
                node_mentions[from_node] += 1
            if to_node not in unique_nodes:
                unique_nodes.add(to_node)
                node_mentions[to_node] = 1
            else:
                node_mentions[to_node] += 1

            edge = (from_node, to_node)
            if edge in repeated_edges:
                repeated_edges[edge] += 1
            else:
                repeated_edges[edge] = 1

            data.add(edge)

    return data, node_mentions, repeated_edges



def plot_node_mentions_histogram(node_mentions, filtered):
    filtered_mentions_values = [count for count in node_mentions.values() if count > filtered]
    filtered_unique_mentions = np.unique(filtered_mentions_values)

    plt.figure(figsize=(8, 6))
    plt.bar(filtered_unique_mentions, [filtered_mentions_values.count(m) for m in filtered_unique_mentions], width=0.8, color='g', edgecolor='k')
    plt.title('Гістограма частот кількості згадувань вершин (після фільтрації)')
    plt.xlabel('Кількість згадувань')
    plt.ylabel('Частота')
    plt.grid(axis='y')
    plt.show()

def compute_clustering_coefficients(filtered_nodes, filtered_data):
    node_clustering_coefficients = {}
    for node in filtered_nodes:
        neighbors = [to_node for from_node, to_node in filtered_data if from_node == node or to_node == node]
        neighbor_count = len(neighbors)
        if neighbor_count < 2:
            clustering_coefficient = 0.0
        else:
            edge_count = 0
            for i in range(neighbor_count):
                for j in range(i + 1, neighbor_count):
                    if (neighbors[i], neighbors[j]) in filtered_data or (neighbors[j], neighbors[i]) in filtered_data:
                        edge_count += 1
            clustering_coefficient = 2.0 * edge_count / (neighbor_count * (neighbor_count - 1))
        node_clustering_coefficients[node] = clustering_coefficient

    network_clustering_coefficient = sum(node_clustering_coefficients.values()) / len(filtered_nodes)
    clustering_coefficient_list = [(node, coefficient) for node, coefficient in node_clustering_coefficients.items()]
    possible_edges = len(filtered_nodes) * (len(filtered_nodes) - 1) / 2
    network_density = len(filtered_data) / possible_edges

    return network_clustering_coefficient, clustering_coefficient_list, network_density

def visualize_graph(filtered_data, node_mentions, repeated_edges, network_clustering_coefficient,vizualize):
    dot = graphviz.Digraph(format='svg', engine='fdp')
    dot.attr(rankdir='LR')

    node_counts = {}
    edges_by_node = {}

    for from_node, to_node in filtered_data:
        node_counts[from_node] = node_mentions[from_node]
        node_counts[to_node] = node_mentions[to_node]

        if from_node not in edges_by_node:
            edges_by_node[from_node] = []
        edges_by_node[from_node].append(to_node)

        if to_node not in edges_by_node:
            edges_by_node[to_node] = []
        edges_by_node[to_node].append(from_node)

    node_size = 0.04 * max(node_counts.values())
    max_count = max(node_counts.values())

    for node, count in node_counts.items():
        normalized_count = count / max_count
        color = cm.viridis(normalized_count)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(255 * color[0]), int(255 * color[1]), int(255 * color[2]))
        if hex_color[1:3] > "b2" or hex_color[3:5] > "b2" or hex_color[5:] > "b2":
            dot.node(node, style='filled', fillcolor=hex_color, fontcolor='black', href="https://www.google.ru/search?q=%22"+node+"%22", z='2', priority='1000')
        else:
            dot.node(node, style='filled', fillcolor=hex_color, fontcolor='white', href="https://www.google.ru/search?q=%22"+node+"%22", z='2', priority='1000')

    for from_node, to_node in filtered_data:
        edge_penwidth =  repeated_edges[(from_node, to_node)]
        edge_url = 'https://www.google.com/search?q=' + from_node + '-' + to_node
        dot.edge(from_node, to_node, style='', URL=edge_url, penwidth=str(edge_penwidth), layer='back')

    dot.render('output_graph', view=vizualize)

    return node_counts, edges_by_node 

def print_graph_info(G,sorted_clustering_coefficient_list, node_counts, filtered_data, edges_by_node,network_density,network_clustering_coefficient):
    print(f"Мережевий кластерний коефіцієнт : {network_clustering_coefficient:.4f}")
    print("Список кластерних коефіцієнтів вузлів:")
    for node, coefficient in sorted_clustering_coefficient_list:
        print(f"{node}: {coefficient:.4f}")

    sorted_node_counts = dict(sorted(node_counts.items(), key=lambda item: item[1], reverse=True))
    print(sorted_node_counts)

    print("Кількість вузлів", len(node_counts))
    print("Кількість ребер", len(filtered_data))

    edges_per_node = [len(edges) for edges in edges_by_node.values()]
    average_degree = sum(edges_per_node) / len(edges_per_node)
    print(f"Середня ступінь графа: {average_degree:.4f}")
    
    print(f"Плотність мережі: {network_density:.4f}")
    
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        print(f"Діаметр графа: {diameter}")

        average_shortest_path_length = nx.average_shortest_path_length(G)
        print(f"Середня довжина найкоротших шляхів в графі: {average_shortest_path_length:.4f}")
    else:
        connected_components = list(nx.connected_components(G))
        average_reciprocal_path_lengths = []

        for component in connected_components:
            subgraph = G.subgraph(component)
            reciprocal_path_lengths = []

            for node1 in component:
                for node2 in component:
                    if node1 != node2:
                        try:
                            reciprocal_path_length = 1 / nx.shortest_path_length(subgraph, node1, node2)
                            reciprocal_path_lengths.append(reciprocal_path_length)
                        except nx.NetworkXNoPath:
                            continue

            if reciprocal_path_lengths:
                avg_reciprocal_path_length = sum(reciprocal_path_lengths) / len(reciprocal_path_lengths)
                average_reciprocal_path_lengths.append(avg_reciprocal_path_length)

        if average_reciprocal_path_lengths:
            avg_reciprocal_path = sum(average_reciprocal_path_lengths) / len(average_reciprocal_path_lengths)
            print(f"Середній інверсний шлях для незв'язного графа: {avg_reciprocal_path:.4f}")

def filter_graph_data(data, node_mentions, filtered):
    filtered_nodes = {node for node, count in node_mentions.items() if count > filtered}
    filtered_data = {(from_node, to_node) for from_node, to_node in data if from_node in filtered_nodes and to_node in filtered_nodes}
    if len(filtered_data) < 2:
        print('Неможливо відфільтрувати, граф зникає')
        sys.exit()
    else:
        return filtered_nodes, filtered_data    
    
def get_filter_value():
    try:
        filtered = 2
        return filtered
    except ValueError:
        print("Будь ласка, введіть коректне число.")
        return get_filter_value()

def main():
    file_path = 'data.txt'

    filtered = get_filter_value()

    data, node_mentions, repeated_edges = read_graph_data(file_path)
    filtered_nodes, filtered_data = filter_graph_data(data, node_mentions, filtered)

    plot_node_mentions_histogram(node_mentions, filtered)

    network_clustering_coefficient, clustering_coefficient_list , network_density = compute_clustering_coefficients(filtered_nodes, filtered_data)

    G = nx.Graph()
    G.add_edges_from(filtered_data)

    visualize_graph(filtered_data, node_mentions, repeated_edges, network_clustering_coefficient,False)
    

    sorted_clustering_coefficient_list = sorted(clustering_coefficient_list, key=lambda x: x[1], reverse=True)

    # Ініціалізуємо node_counts перед передачею у функцію:
    node_counts, edges_by_node = visualize_graph(filtered_data, node_mentions, repeated_edges, network_clustering_coefficient, True)

    print_graph_info(G,sorted_clustering_coefficient_list, node_counts, filtered_data, edges_by_node,network_density,network_clustering_coefficient)

if __name__ == "__main__":
    main()
