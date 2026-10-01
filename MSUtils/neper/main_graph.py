from pathlib import Path
import numpy as np

from MSUtils.neper.NeperTessellation import NeperTessellation
from MSUtils.neper.NeperMicrostructure import NeperMicrostructure
from MSUtils.neper.NeperGBErosion import NeperGBErosion
from MSUtils.neper.PolycrystalGraph import PolycrystalGrainGraph, PolycrystalFacetEnhancedGraph, PolycrystalVertexEnhancedGraph

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    Nx, Ny, Nz = 256, 256, 256
    L = (1.0, 1.0, 1.0)
    num_grains = 64
    interface_thickness = 6 * L[0] / Nx
    h5_filename = Path("data/neper_microstructures.h5")

    tesr_directory = Path("data/neper")
    neper_executable = _PROJECT_ROOT / ".pixi/envs/neper/bin/neper"

    name = "periodic_voronoi"
    parameters = {
        "num_grains": num_grains,
        "morphology": "voronoi",
        "periodicity": "all",
        "orientation": "random",
        "crystal_symmetry": "mmm",
    }
    # name = "cubes_64"
    # parameters = {
    #     "num_grains": num_grains,
    #     "morphology": "cube",
    #     "periodicity": "all",
    #     "orientation": "random",
    #     "crystal_symmetry": "mmm",
    # }
    seed = 1

    microstructure = NeperMicrostructure(
        tesr_directory / name,
        **parameters,
        neper_executable=neper_executable,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        L=L,
        seed=seed,
    )

    microstructure.write_h5(h5_filename, name)

    # erosion = NeperGBErosion(microstructure, interface_thickness)
    
    # # Export not required
    # erosion.write_h5(
    #     h5_filename,
    #     name,
    #     save_normals=True,
    #     save_orientations=False,
    # )

    tess = NeperTessellation(f"{_PROJECT_ROOT}/data/neper/{name}.tess")

    graph = PolycrystalGrainGraph.from_tess(tess, [], [])
    graph.graph_stats()
    # graph.visualize_3d(Path(f"{_PROJECT_ROOT}/data/{name}_grain_graph"), node_feature="type")
    grp = "graph_grains"
    graph.write_h5(Path(f"{_PROJECT_ROOT}/data/{name}_graph"), grp="graph_grains", 
                   node_attr=[["position", "type"]], edge_attr=["distance", "distance"], 
                   export_stats=True)
    graph.write_xdmf(f"{_PROJECT_ROOT}/data/{name}_graph_{grp}.xdmf", 
                        node_values={'type': None, 'test': np.zeros(graph.G.number_of_nodes())}, 
                        edge_values={'test_edges': np.ones(int(graph.G.number_of_edges()/2))})

    # graph = PolycrystalFacetEnhancedGraph.from_tess(tess, [], ["distance"])
    # graph.graph_stats()
    # # graph.assign_eroded_material_indices(erosion)
    # # graph.visualize_3d(Path(f"{_PROJECT_ROOT}/data/{name}_facet_graph"), node_feature="type")
    # graph.write_h5(Path(f"{_PROJECT_ROOT}/data/{name}_graph"), grp="graph_facet_enhanced", export_stats=True)
    
    graph = PolycrystalVertexEnhancedGraph.from_tess(tess, [], ["distance"])
    graph.graph_stats()
    # graph.visualize_3d(Path(f"{_PROJECT_ROOT}/data/{name}_vertex_graph"), node_feature="type")
    grp = "graph_vertex_enhanced"
    graph.write_h5(Path(f"{_PROJECT_ROOT}/data/{name}_graph"), grp=grp, 
                   node_attr=[["position", "type"], ["type"], ["position"]], edge_attr=["distance"],
                   export_stats=True)
    graph.write_xdmf(f"{_PROJECT_ROOT}/data/{name}_graph_{grp}.xdmf", 
                        node_values={'type': None, 'test': np.zeros(graph.G.number_of_nodes())}, 
                        edge_values={'test_edges': np.ones(int(graph.G.number_of_edges()/2))})


if __name__ == "__main__":
    main()