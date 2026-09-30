from typing import Self
import numpy as np
import h5py
from pathlib import Path
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx
from collections import defaultdict
import copy

from MSUtils.neper.NeperTessellation import NeperTessellation
from MSUtils.neper.NeperGBErosion import NeperGBErosion


def _gname(grain_id: int, shift=False):
    return f'g_{grain_id-int(shift):05}'

def _vname(vertex_id: int, shift=False):
    return f'v_{vertex_id-int(shift):05}'

def _fname(facet_id: int, shift=False):
    return f'f_{facet_id-int(shift):05}'


class PolycrystalGraph:
    def __init__(
        self,
    ) -> Self:
        self.G = nx.MultiDiGraph()

    @classmethod
    def from_tess(
        cls,
        tessellation: NeperTessellation,
        node_features: list[str],
        edge_features: list[str],
    ):
        graph = cls()
    
        # Pass deep-copy to avoid altered tessellation
        graph._construct_topology(copy.deepcopy(tessellation))

        graph.node_feature_labels = node_features
        graph._add_node_features()

        graph.edge_feature_labels = edge_features
        graph._add_edge_features()

        return graph

    @classmethod
    def from_data(
        cls,
        filepath: Path,
        grp: str,
    ):
        with h5py.File(filepath, 'r') as h5_file:
            grp = h5_file[grp]
            x_grain     = grp['node_features_0'][:]
            if 'node_features_1' in grp.keys():
                x_facet     = grp['node_features_1'][:]
            else:
                x_facet = []
            if 'node_features_2' in grp.keys():
                x_vertex    = grp['node_features_2'][:]
            else:
                x_vertex = []
            edge_index  = grp['edge_index'][:]
            #edge_attr   = grp['edge_features'][:]

        graph = cls()

        n_nodes = len(x_grain) + len(x_facet) + len(x_vertex)

        if (len(x_facet) > 0) and (len(x_vertex) > 0):
            x = np.vstack((x_grain[:,1:4], x_facet[:,1:4], x_vertex[:,1:4]))
        elif len(x_facet) > 0:
            x = np.vstack((x_grain[:,1:4], x_facet[:,1:4]))
        else:
            x = x_grain[:,1:4]

        for i in range(n_nodes):
            graph.G.add_node(
                i,
                position=x[i]
            )

        mask = edge_index[0] < edge_index[1]
        edge_index_unique = edge_index[:, mask]
        #_edge_attr = edge_attr[mask]
        graph.G.add_edges_from(edge_index_unique.T)

        return graph


    def _construct_topology(self, tessellation: NeperTessellation):
        raise NotImplementedError

    def _add_node_features(self):
        raise NotImplementedError

    def _add_edge_features(self):
        raise NotImplementedError
    
    def graph_stats(
        self
    ):
        degrees = [d for _, d in self.G.degree()]
        print("=========================")
        print("Graph statisics:")
        print(f"  {type(self)}")
        print("=========================")
        print(f"Nodes:       {self.G.number_of_nodes():,}")
        print(f"Edges:       {self.G.number_of_edges():,}")
        print(f"Density:     {nx.density(self.G):.4f}")
        #print(f"Connected:   {nx.is_connected(self.G)}")
        #print(f"Components:  {nx.number_connected_components(self.G)}")
        print(f"Avg degree:  {sum(degrees) / len(degrees):.2f}")
        print(f"Max degree:  {max(degrees):,}")
        print(f"Min degree:  {min(degrees):,}\n")

    def visualize_3d(
        self,
        filepath: Path,
        node_feature: str | None = None,
        node_data = None,
    ):
        """Visualize a 3D NetworkX graph as a rotating transparent GIF.

        Each node of the NetworkX graph must have a 3D ``"position"`` attribute.

        Parameters
        ----------
        filepath:
            Output path. The GIF is written using the same path with a ``.gif`` suffix.
        node_feature:
            Optional node attribute containing a scalar value used to color the nodes.
        node_data:
            Optional scalar node data used to color the nodes if node_feature is set to None.
        """

        # --------------------------------------------------
        # Extract graph data
        # --------------------------------------------------

        nodes = list(self.G.nodes)

        node_positions = np.asarray(
            [self.G.nodes[node]["position"] for node in nodes],
            dtype=float,
        )

        if node_positions.ndim != 2 or node_positions.shape[1] != 3:
            raise ValueError(
                'Each node must have a 3D "position" attribute.'
            )

        node_to_idx = {node: i for i, node in enumerate(nodes)}

        edges = [
            (node_to_idx[u], node_to_idx[v])
            for u, v, _ in self.G.edges
        ]

        # --------------------------------------------------
        # Node colors
        # --------------------------------------------------

        values = None
        if node_feature is not None:
            try:
                values = np.asarray(
                    [self.G.nodes[node][node_feature] for node in nodes],
                    dtype=float,
                )
            except KeyError as exc:
                raise KeyError(
                    f'Node attribute "{node_feature}" is missing.'
                ) from exc
        elif node_data is not None:   
            assert node_data.shape == (self.G.number_of_nodes(),)

            values = node_data

        if values is not None:
            norm = Normalize(
                vmin=values.min(),
                vmax=values.max(),
            )
            cmap = plt.colormaps["jet"]

        # --------------------------------------------------
        # Plot limits
        # --------------------------------------------------

        mins = node_positions.min(axis=0)
        maxs = node_positions.max(axis=0)

        span = np.maximum(maxs - mins, 1e-6)
        padding = 0.05 * span

        lower = mins - padding
        upper = maxs + padding

        # --------------------------------------------------
        # Create figure ONCE
        # --------------------------------------------------

        fig = plt.figure(
            figsize=(8, 8),
            dpi=100,
            facecolor="none",
        )

        ax = fig.add_subplot(
            111,
            projection="3d",
            facecolor="none",
        )

        ax.set_axis_off()

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])

        # --------------------------------------------------
        # Draw edges ONCE
        # --------------------------------------------------

        for i, j in edges:
            ax.plot(
                node_positions[[i, j], 0],
                node_positions[[i, j], 1],
                node_positions[[i, j], 2],
                color="gray",
                alpha=0.2,
                linewidth=0.3,
            )

        # --------------------------------------------------
        # Draw nodes ONCE
        # --------------------------------------------------

        if values is not None:
            ax.scatter(
                node_positions[:, 0],
                node_positions[:, 1],
                node_positions[:, 2],
                s=25,
                c=values,
                cmap=cmap,
                norm=norm,
                depthshade=True,
            )
        else:
            ax.scatter(
                node_positions[:, 0],
                node_positions[:, 1],
                node_positions[:, 2],
                s=25,
                color="white",
                depthshade=True,
            )

        # --------------------------------------------------
        # Render frames
        # --------------------------------------------------

        frames = []

        for angle in range(0, 360, 4):
            ax.view_init(
                elev=25,
                azim=angle,
            )

            buffer = io.BytesIO()

            fig.savefig(
                buffer,
                format="png",
                transparent=True,
                #bbox_inches="tight",
                pad_inches=0,
            )

            buffer.seek(0)

            frame = Image.open(buffer).convert("RGBA").copy()
            frames.append(frame)

            buffer.close()

        plt.close(fig)

        # --------------------------------------------------
        # Convert frames to GIF palette
        # --------------------------------------------------

        palette_frames = []

        for frame in frames:
            palette = frame.convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
            )

            alpha = frame.getchannel("A")

            palette.info["transparency"] = 0

            transparent_mask = alpha.point(
                lambda a: 255 if a == 0 else 0
            )

            palette.paste(
                0,
                mask=transparent_mask,
            )

            palette_frames.append(palette)

        # --------------------------------------------------
        # Save GIF
        # --------------------------------------------------

        output_path = filepath.with_suffix(".gif")

        palette_frames[0].save(
            output_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=50,
            loop=0,
            disposal=2,
            transparency=0,
        )

    def write_graph_xdmf(
        self,
        filename,
        node_values,
        edge_values,
    ):
        """
        Write a graph to an XDMF file.

        Parameters
        ----------
        filename : str
            Output filename, e.g. "graph.xdmf".

        node_values : array-like, shape (N,)
            Scalar value associated with each node.

        edge_values : array-like, shape (M,)
            Scalar value associated with each edge.
        """

        positions = np.array([data["position"] for _, data in self.G.nodes(data=True)])
        edges = list(self.G.edges(data=True))

        connectivity = np.array(
            [(u, v) for u, v, data in edges],
            dtype=int
        )
        node_values = np.asarray(node_values, dtype=float)
        edge_values = np.asarray(edge_values, dtype=float)

        # ------------------------------------------------------------
        # Validate input
        # ------------------------------------------------------------

        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError(
                "positions must have shape (N, 2) or (N, 3)"
            )

        n_nodes = positions.shape[0]

        if connectivity.ndim != 2 or connectivity.shape[1] != 2:
            raise ValueError(
                "connectivity must have shape (M, 2)"
            )

        n_edges = connectivity.shape[0]

        if len(node_values) != n_nodes:
            raise ValueError(
                f"node_values has length {len(node_values)}, "
                f"but there are {n_nodes} nodes"
            )

        if len(edge_values) != n_edges:
            raise ValueError(
                f"edge_values has length {len(edge_values)}, "
                f"but there are {n_edges} edges"
            )

        if np.any(connectivity < 0) or np.any(connectivity >= n_nodes):
            raise ValueError(
                "connectivity contains invalid node indices"
            )

        # # ------------------------------------------------------------
        # # Convert 2D coordinates to 3D
        # # ------------------------------------------------------------

        # if positions.shape[1] == 2:
        #     positions = np.column_stack(
        #         [positions, np.zeros(n_nodes)]
        #     )

        # ------------------------------------------------------------
        # Convert arrays to XDMF text
        # ------------------------------------------------------------

        coordinates_text = "\n".join(
            f"{x:.16g} {y:.16g} {z:.16g}"
            for x, y, z in positions
        )

        connectivity_text = "\n".join(
            f"{i} {j}"
            for i, j in connectivity
        )

        node_values_text = "\n".join(
            f"{v:.16g}"
            for v in node_values
        )

        edge_values_text = "\n".join(
            f"{v:.16g}"
            for v in edge_values
        )

        # ------------------------------------------------------------
        # XDMF document
        # ------------------------------------------------------------

        xdmf = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">

<Xdmf Version="3.0">
  <Domain>

    <Grid Name="Graph" GridType="Uniform">

      <!-- =====================================================
           Graph connectivity
           ===================================================== -->

      <Topology
          TopologyType="PolyLine"
          NumberOfElements="{n_edges}"
          NodesPerElement="2">

        <DataItem
            Format="XML"
            NumberType="Int"
            Dimensions="{n_edges} 2">
{connectivity_text}
        </DataItem>

      </Topology>


      <!-- =====================================================
           Node coordinates
           ===================================================== -->

      <Geometry GeometryType="XYZ">

        <DataItem
            Format="XML"
            NumberType="Float"
            Precision="8"
            Dimensions="{n_nodes} 3">
{coordinates_text}
        </DataItem>

      </Geometry>


      <!-- =====================================================
           Scalar value associated with each node
           ===================================================== -->

      <Attribute
          Name="node_value"
          AttributeType="Scalar"
          Center="Node">

        <DataItem
            Format="XML"
            NumberType="Float"
            Precision="8"
            Dimensions="{n_nodes}">
{node_values_text}
        </DataItem>

      </Attribute>


      <!-- =====================================================
           Scalar value associated with each edge
           ===================================================== -->

      <Attribute
          Name="edge_value"
          AttributeType="Scalar"
          Center="Cell">

        <DataItem
            Format="XML"
            NumberType="Float"
            Precision="8"
            Dimensions="{n_edges}">
{edge_values_text}
        </DataItem>

      </Attribute>

    </Grid>

  </Domain>
</Xdmf>
"""

        with open(filename, "w") as f:
            f.write(xdmf)

    def write_h5(
        self,
        filename: Path,
        grp: str,
        export_stats: bool = False,
    ) -> None:
        nodes = list(self.G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Node features
        nodes_by_type = defaultdict(list)
        for n, attrs in self.G.nodes(data=True):
            nodes_by_type[attrs.get("type")].append(n)

        X = dict()
        for t in self.node_schemas:
            nodes_t = nodes_by_type[int(t)]
            X[int(t)] = np.zeros((len(nodes_t), sum(self.node_schemas[t].values())))
            for i_n, n in enumerate(nodes_t):
                idx = 0
                for f in self.node_schemas[t].keys():

                    X[int(t)][i_n, idx:idx+self.node_schemas[t][f]] = self.G.nodes[n].get(f, 0)
                    idx += self.node_schemas[t][f]

        # Edges, both directions
        edge_features = {key for _, _, attrs in self.G.edges(data=True) for key in attrs}
        edges = []
        E = []

        for u, v, data in self.G.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            features = np.concatenate([
                np.atleast_1d(data.get(f, 0)).ravel()
                for f in edge_features
            ])

            edges.extend([(i, j)])
            E.extend([features])

        edge_index = np.asarray(edges, dtype=np.int64).T
        E = np.asarray(E, dtype=np.float32)

        with h5py.File(filename.with_suffix('.h5'), "a") as h5_file:
            compression_opts = 6

            grp = h5_file.require_group(grp)

            grp["edge_index"] = edge_index
            for t in self.node_schemas:
                dset = grp.create_dataset(
                    f'node_features_{t}',
                    data=X[int(t)],
                    dtype='f8',
                    compression='gzip',
                    compression_opts=compression_opts
                )
                dset.attrs['n_nodes_type'] = len(nodes_by_type[int(t)])
                dset.attrs['order'] = str(self.node_schemas[t])

            dset = grp.create_dataset(
                'edge_features',
                data=E,
                dtype='f8',
                compression='gzip',
                compression_opts=compression_opts
            )
            dset.attrs['n_edges'] = self.G.number_of_edges()
            dset.attrs['order'] = str(edge_features)

            dset = grp.create_dataset(
                'node_ids',
                data=np.asarray(nodes, dtype='S')
            )
            dset.attrs['n_nodes'] = self.G.number_of_nodes()


            if export_stats:
                degrees = [d for _, d in self.G.degree()]
                grp_stats = grp.require_group("stats")

                grp_stats.create_dataset(
                    'density',
                    data=nx.density(self.G),
                    dtype='f8'
                )

                dset = grp_stats.create_dataset(
                    'degrees',
                    data=degrees,
                    dtype='f8'
                )
                dset.attrs['average'] = sum(degrees) / len(degrees)
                dset.attrs['min'] = min(degrees)
                dset.attrs['max'] = max(degrees)

        
class PolycrystalGrainGraph(PolycrystalGraph):

    def _construct_topology(
        self, 
        tessellation: NeperTessellation
    ):

        # ---------------------------------------------------------------
        # Add nodes
        # ---------------------------------------------------------------

        # Add one node per grain (position given by seed in tessellation)
        self.G.add_nodes_from(
            (_gname(grain_id, shift=True), 
             {"position": grain_data, 
              "type": 0,
              "boundary": 0,
              "mat_idx": grain_id})
            for grain_id, grain_data in tessellation.seeds.items()
        )

        # ---------------------------------------------------------------
        # Add edges
        # ---------------------------------------------------------------

        # Connect grains that share a facet
        for facet_id, grains in tessellation.facet_grains.items():
            if len(grains) == 2:
                _grain_i = _gname(grains.pop(), shift=True)
                _grain_j = _gname(grains.pop(), shift=True)

                self.G.add_edge(
                    _grain_i, _grain_j,
                    # Compute standard distance (no un-wrapping of periodicity needed)
                    distance=(
                        np.array(self.G.nodes[_grain_j]["position"]) - 
                        np.array(self.G.nodes[_grain_i]["position"])),
                    area=tessellation._get_facet_surface(facet_id)
                    )
                self.G.add_edge(
                    _grain_j, _grain_i,
                    # Compute standard distance (no un-wrapping of periodicity needed)
                    distance=(
                        np.array(self.G.nodes[_grain_i]["position"]) - 
                        np.array(self.G.nodes[_grain_j]["position"])),
                    area=tessellation._get_facet_surface(facet_id)
                    )

        # Wrap around periodic boundary
        for secondary_node_idx, data in tessellation.periodicity["facets"].items():
            assert len( tessellation.facet_grains[data["primary"]]) == 1
            _grain_i = _gname(tessellation.facet_grains[data["primary"]].pop(), shift=True)
            assert len(tessellation.facet_grains[secondary_node_idx]) == 1
            _grain_j = _gname(tessellation.facet_grains[secondary_node_idx].pop(), shift=True)

            self.G.add_edge(
                _grain_i, _grain_j,
                # Periodic unwrapping via given shift from tessellation
                distance=(
                    np.array(self.G.nodes[_grain_j]["position"]) -
                    np.array(self.G.nodes[_grain_i]["position"]) -
                    np.array(data["shift"])
                    ),
                area=tessellation._get_facet_surface(data["primary"])
            )
            self.G.add_edge(
                _grain_j, _grain_i,
                # Periodic unwrapping via given shift from tessellation
                distance=(
                    np.array(self.G.nodes[_grain_i]["position"]) -
                    np.array(self.G.nodes[_grain_j]["position"]) +
                    np.array(data["shift"])
                    ),
                area=tessellation._get_facet_surface(data["primary"])
            )

        self.node_schemas = {
            "0":  {"type": 1, "position": 3, "boundary": 1},
        }

    def _add_node_features(
        self,
    ) -> None:
        ...

    def _add_edge_features(
        self,
    ) -> None:
        ...

class PolycrystalFacetEnhancedGraph(PolycrystalGraph):

    def _construct_topology(
        self,
        tessellation: NeperTessellation
    ) -> None:

        # ---------------------------------------------------------------
        # Add nodes
        # ---------------------------------------------------------------

        # Add one node per grain (position given by seed in tessellation)
        self.G.add_nodes_from(
            (_gname(grain_id, shift=True), 
                {"position": grain_data, 
                 "type": 0,
                 "boundary": 0})
            for grain_id, grain_data in tessellation.seeds.items()
        )
        # Add one node per facet (compute centroid for position)
        self.G.add_nodes_from(
            (_fname(facet_id, shift=True), 
                {"position": np.mean([tessellation.vertices[vertex_id] for vertex_id in facet['vertices']], axis=0),
                 "type": 1,
                 "boundary": 0})
            for facet_id, facet in tessellation.facets.items()
        )

        # ---------------------------------------------------------------
        # Add edges
        # ---------------------------------------------------------------

        # Connect facets to grains they separate
        for facet_id, grains in tessellation.facet_grains.items():
            _facet_i = _fname(facet_id, shift=True)
            for _ in range(len(grains)):
                _grain_j = _gname(grains.pop(), shift=True)
                dist = np.array(self.G.nodes[_grain_j]["position"]) - np.array(self.G.nodes[_facet_i]["position"])
                self.G.add_edge(
                    _facet_i, _grain_j,
                    distance=dist,
                    length=np.linalg.norm(dist),
                    area=tessellation._get_facet_surface(facet_id) * 0.01
                    )
                self.G.add_edge(
                    _grain_j, _facet_i,
                    distance=-dist,
                    length=np.linalg.norm(dist),
                    area=tessellation._get_facet_surface(facet_id) * 0.01
                    )

        # Connect facets that share an edge
        for edge_id, facets in tessellation.edge_facets.items():
            _facet_i = _fname(facets.pop(), shift=True)
            _facet_j = _fname(facets.pop(), shift=True)
            # Identify the vertices forming the edge
            (_vertex_k, _vertex_l) = tessellation.edges[edge_id]
            _edge_pos = (np.array(tessellation.vertices[_vertex_k]) + np.array(tessellation.vertices[_vertex_l])) / 2
            self.G.add_edge(
                _facet_i, _facet_j,
                # Construct lenght via edge separating the facets
                distance=np.array(self.G.nodes[_facet_j]["position"]) -
                    np.array(self.G.nodes[_facet_i]["position"]),
                length=np.linalg.norm(np.array(self.G.nodes[_facet_j]["position"]) - _edge_pos) +
                    np.linalg.norm(np.array(self.G.nodes[_facet_i]["position"]) - _edge_pos),
                area=tessellation._get_edge_length(edge_id)
            )
            self.G.add_edge(
                _facet_j, _facet_i,
                # Construct lenght via edge separating the facets
                distance=np.array(self.G.nodes[_facet_i]["position"]) -
                    np.array(self.G.nodes[_facet_j]["position"]),
                length=np.linalg.norm(np.array(self.G.nodes[_facet_j]["position"]) - _edge_pos) +
                    np.linalg.norm(np.array(self.G.nodes[_facet_i]["position"]) - _edge_pos),
                area=tessellation._get_edge_length(edge_id)
            )

        # Account for periodic boundary by merging corresponding facet nodes to primary instance
        mapping = {}
        for secondary_node_idx, data in tessellation.periodicity["facets"].items():
            # Make sure that nodes to be merged do not have common edges (might alter attributes)
            assert len([x for x in list(self.G.neighbors(_fname(data["primary"], shift=True))) 
                        if x in list(self.G.neighbors(_fname(secondary_node_idx, shift=True)))]) == 0
            mapping[_fname(secondary_node_idx, shift=True)] = _fname(data["primary"], shift=True)
        nx.relabel_nodes(self.G, mapping, copy=False)

        # ---------------------------------------------------------------
        # Mark boundary nodes for periodic tessellation
        # ---------------------------------------------------------------

        # Tag periodicity relations of facets
        # nx.set_node_attributes(self.G, [np.nan, np.nan, np.nan], "boundary_shift")
        # for secondary_node_idx, data in tessellation.periodicity["facets"].items():
        #     self.G.nodes[_fname(secondary_node_idx)]["boundary"] = 2
        #     self.G.nodes[_fname(data["primary"])]["boundary"] = 1
        #     self.G.nodes[_fname(data["primary"])]["boundary_shift"] = data["shift"]

        self.node_schemas = {
            "0":  {"type": 1, "position": 3, "boundary": 1},
            "1":  {"type": 1, "position": 3, "boundary": 1, "boundary_shift": 3}
        }

    def assign_eroded_material_indices(
        self,
        erosion: NeperGBErosion,
    ) -> None:
        for tag, (_, grain_i, grain_j) in erosion.ridge_metadata.items():
            # First, identify all facet nodes that have the respective grain pair as neighbors
            common_neighbors = nx.common_neighbors(self.G, _gname(grain_i), _gname(grain_j))
            if len(common_neighbors) == 1:
                self.G.nodes[common_neighbors.pop()]["mat_idx"] = tag
            else:
                assert False
                # Mutiple facet nodes separate the same grain pair
                # TODO
                

        for node, data in self.G.nodes(data=True):
            print(node, data.get("mat_idx"), list(self.G.neighbors(node)))

            

    def _add_node_features(
        self,
    ) -> None:
        ...
        # Orientation information
        # for node, data in self.G.nodes(data=True):
        #     if data["type"] == 1:
        #         node["orientation"] = 


    def _add_edge_features(
        self,
    ) -> None:
        ...

class PolycrystalVertexEnhancedGraph(PolycrystalGraph):

    def _construct_topology(
        self,
        tessellation: NeperTessellation
    ) -> None:
        
        # ---------------------------------------------------------------
        # Add nodes
        # ---------------------------------------------------------------

        # Add one node per grain (position given by seed in tessellation)
        self.G.add_nodes_from(
            (_gname(grain_id, shift=True), 
             {"position": grain_data, 
              "type": 0,
              "boundary": 0})
            for grain_id, grain_data in tessellation.seeds.items()
        )
        # Add one node per facet (compute centroid for position)
        self.G.add_nodes_from(
            (_fname(facet_id, shift=True), 
             {"position": np.mean([tessellation.vertices[vertex_id] for vertex_id in facet['vertices']], axis=0),
              "type": 1,
              "boundary": 0})
            for facet_id, facet in tessellation.facets.items()
        )
        # Add one node per vertex (position given in tessellation)
        self.G.add_nodes_from(
            (_vname(vertex_id, shift=True), 
             {"position": vertex_data, 
              "type": 2,
              "boundary": 0})
            for vertex_id, vertex_data in tessellation.vertices.items()
        )

        # ---------------------------------------------------------------
        # Add edges
        # ---------------------------------------------------------------

        # Connect facets to grains they separate
        for facet_id, grains in tessellation.facet_grains.items():
            for _ in range(len(grains)):
                _facet_i = _fname(facet_id, shift=True)
                _grain_j = _gname(grains.pop(), shift=True)
                self.G.add_edge(
                    _facet_i, _grain_j,
                    distance=(
                        np.array(self.G.nodes[_grain_j]["position"]) -
                        np.array(self.G.nodes[_facet_i]["position"])
                    )
                )
                self.G.add_edge(
                    _grain_j, _facet_i,
                    distance=(
                        np.array(self.G.nodes[_facet_i]["position"]) -
                        np.array(self.G.nodes[_grain_j]["position"])
                    )
                )

        # Connect vertices that form an edge
        for edge_id, (v1, v2) in tessellation.edges.items():
            _vertex_i = _vname(v1, shift=True)
            _vertex_j = _vname(v2, shift=True) 
            self.G.add_edge(
                _vertex_i, _vertex_j,
                distance=(
                    np.array(self.G.nodes[_vertex_j]["position"]) -
                    np.array(self.G.nodes[_vertex_i]["position"])
                )
            )
            self.G.add_edge(
                _vertex_j, _vertex_i,
                distance=(
                    np.array(self.G.nodes[_vertex_i]["position"]) -
                    np.array(self.G.nodes[_vertex_j]["position"])
                )
            )

        # Connect vertices to the centroid of the facet they form
        for facet_id, facet in tessellation.facets.items():
            for vertex_id in facet['vertices']:
                _facet_i = _fname(facet_id, shift=True)
                _vertex_j = _vname(vertex_id, shift=True)
                self.G.add_edge(
                    _facet_i, _vertex_j,
                    distance=(
                        np.array(self.G.nodes[_vertex_j]["position"]) -
                        np.array(self.G.nodes[_facet_i]["position"])
                    )
                )
                self.G.add_edge(
                    _vertex_j, _facet_i,
                    distance=(
                        np.array(self.G.nodes[_facet_i]["position"]) -
                        np.array(self.G.nodes[_vertex_j]["position"])
                    )
                ) 

        # ---------------------------------------------------------------
        # Mark boundary nodes for periodic tessellation
        # ---------------------------------------------------------------

        # Tag periodicity relations of vertices
        # nx.set_node_attributes(self.G, [np.nan, np.nan, np.nan], "boundary_shift")
        # for secondary_node_idx, data in tessellation.periodicity["vertices"].items():
        #     self.G.nodes[_vname(secondary_node_idx)]["boundary"] = 2
        #     self.G.nodes[_vname(data["primary"])]["boundary"] = 1
        #     self.G.nodes[_vname(data["primary"])]["boundary_shift"] = data["shift"]

        # # Tag periodicity relations of facets
        # nx.set_node_attributes(self.G, [np.nan, np.nan, np.nan], "boundary_shift")
        # for secondary_node_idx, data in tessellation.periodicity["facets"].items():
        #     self.G.nodes[_fname(secondary_node_idx)]["boundary"] = 2
        #     self.G.nodes[_fname(data["primary"])]["boundary"] = 1
        #     self.G.nodes[_fname(data["primary"])]["boundary_shift"] = data["shift"]

        # Account for periodic boundary by merging corresponding facet nodes to primary instance
        mapping = {}
        for secondary_node_idx, data in tessellation.periodicity["facets"].items():
            # Make sure that nodes to be merged do not have common edges (might alter attributes)
            assert len([x for x in list(self.G.neighbors(_fname(data["primary"], shift=True))) 
                        if x in list(self.G.neighbors(_fname(secondary_node_idx, shift=True)))]) == 0
            mapping[_fname(secondary_node_idx, shift=True)] = _fname(data["primary"], shift=True)
        for secondary_node_idx, data in tessellation.periodicity["vertices"].items():
            # Make sure that nodes to be merged do not have common edges (might alter attributes)
            assert len([x for x in list(self.G.neighbors(_vname(data["primary"], shift=True))) 
                        if x in list(self.G.neighbors(_vname(secondary_node_idx, shift=True)))]) == 0
            mapping[_vname(secondary_node_idx, shift=True)] = _vname(data["primary"], shift=True)
        nx.relabel_nodes(self.G, mapping, copy=False)

        self.node_schemas = {
            "0":  {"type": 1, "position": 3, "boundary": 1},
            "1":  {"type": 1, "position": 3, "boundary": 1, "boundary_shift": 3},
            "2":  {"type": 1, "position": 3, "boundary": 1, "boundary_shift": 3}
        }

    def _add_node_features(
        self,
    ) -> None:
        ...

    def _add_edge_features(
        self,
    ) -> None:
        ...
