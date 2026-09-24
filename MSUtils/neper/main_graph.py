from pathlib import Path

from MSUtils.neper.NeperTessellation import NeperTessellation
from MSUtils.neper.PolycrystalGraph import PolycrystalGrainGraph, PolycrystalFacetEnhancedGraph, PolycrystalVertexEnhancedGraph

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    name = "periodic_voronoi"
    tess = NeperTessellation(f"{_PROJECT_ROOT}/data/neper/{name}.tess")

    graph = PolycrystalGrainGraph(tess, [], [])
    graph.graph_stats()
    graph.visualize_3d(Path(f"{_PROJECT_ROOT}/data/{name}_grain_graph"), node_feature="type")
    graph.write_h5(Path(f"{_PROJECT_ROOT}/data/{name}_graph"), grp="graph_grains", export_stats=True)

    graph = PolycrystalFacetEnhancedGraph(tess, [], ["distance"])
    graph.graph_stats()
    graph.visualize_3d(Path(f"{_PROJECT_ROOT}/data/{name}_facet_graph"), node_feature="type")
    graph.write_h5(Path(f"{_PROJECT_ROOT}/data/{name}_graph"), grp="graph_facet_enhanced", export_stats=True)

    graph = PolycrystalVertexEnhancedGraph(tess, [], ["distance"])
    graph.graph_stats()
    graph.visualize_3d(Path(f"{_PROJECT_ROOT}/data/{name}_vertex_graph"), node_feature="type")
    graph.write_h5(Path(f"{_PROJECT_ROOT}/data/{name}_graph"), grp="graph_vertex_enhanced", export_stats=True)


if __name__ == "__main__":
    main()