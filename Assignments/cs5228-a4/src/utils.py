import networkx as nx
from matplotlib import pyplot as plt




def plot_mrt_graph(G, df):
    fixed_positions = {}

    for idx, n in enumerate(G.nodes):
        node = str(n).lower()
        row = df.loc[df.name.str.lower() == node].iloc[0]
        lat, lng = row.latitude, row.longitude
        #print(idx)
        fixed_positions[n] = (lng, lat)

    fixed_nodes = fixed_positions.keys()

    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)  
    
    plt.figure()
    plt.axis('equal')
    nx.draw(G, pos=pos, node_size=100, with_labels=True, font_size=8) 
    plt.show()