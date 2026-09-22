from pathlib import Path

from MSUtils.neper.NeperTessellation import NeperTessellation
from MSUtils.neper.PolycrystalGraph import PolycrystalGrainGraph, PolycrystalFacetEnhancedGraph, PolycrystalVertexEnhancedGraph


def main():
    name = "diamond"
    tess = NeperTessellation(f"/home/scholz/workspace/MSUtils/data/neper/{name}.tess")

    graph = PolycrystalGrainGraph(tess, [], [])
    graph.graph_stats()
    graph.visualize_3d(Path(f"./data/grain_graph_{name}"), node_feature="boundary")
    graph.write_h5(Path(f"./data/grain_graph_{name}"), node_features=["position", "boundary"], edge_features=[])

    graph = PolycrystalFacetEnhancedGraph(tess, [], [])
    graph.graph_stats()
    graph.visualize_3d(Path(f"./data/facet_graph_{name}"), node_feature="boundary")
    graph.write_h5(Path(f"./data/facet_graph_{name}"), node_features=["position", "boundary"], edge_features=[])

    graph = PolycrystalVertexEnhancedGraph(tess, [], [])
    graph.graph_stats()
    graph.visualize_3d(Path(f"./data/vertex_graph_{name}"), node_feature="boundary")
    graph.write_h5(Path(f"./data/vertex_graph_{name}"), node_features=["position", "boundary"], edge_features=[])


if __name__ == "__main__":
    main()