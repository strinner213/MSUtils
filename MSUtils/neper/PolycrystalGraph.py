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
        periodicity: bool = True,
    ) -> Self:
        self.G = nx.MultiDiGraph()
        self.periodicity = periodicity

    @classmethod
    def from_tess(
        cls,
        tessellation: NeperTessellation,
        node_features: list[str],
        edge_features: list[str],
    ):
        graph = cls(tessellation.periodicity)
    
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
        self,
    ) -> None:
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

    def attribute_names(
        self,
    ) -> None:
        node_attrs = set().union(*(d.keys() for _, d in self.G.nodes(data=True)))
        edge_attrs = set().union(*(d.keys() for _, _, d in self.G.edges(data=True)))
        return node_attrs, edge_attrs

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

    def write_xdmf(
        self,
        filename: dict,
        node_values: dict,
        edge_values: dict,
    ):
        """
        Write a graph to an XDMF file.

        Parameters
        ----------
        filename : str
            Output filename, e.g. "graph.xdmf".

        node_values : dict
            Each component consists of label (str) and data (scalar value associated with each node).
            If the label is an attribute on the graph the graph's attribute data is used. Then None can be provided instead of data.

        edge_values : dict
            Each component consists of label (str) and data (scalar value associated with each node).
            If the label is an attribute on the graph the graph's attribute data is used. Then None can be provided instead of data.
        """

        positions = np.array([data["position"] for _, data in self.G.nodes(data=True)])
        edges = list(self.G.edges(data=True))

        # Map node names -> integers
        node_to_int = {node: i for i, node in enumerate(self.G.nodes())}
        connectivity = np.array(
            [(node_to_int[u], node_to_int[v]) for u, v, data in edges],
            dtype=int
        )
        mask = connectivity[:,0] < connectivity[:,1]
        connectivity = connectivity[mask,:]


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

        if np.any(connectivity < 0) or np.any(connectivity >= n_nodes):
            raise ValueError(
                "connectivity contains invalid node indices"
            )

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
"""
        
        node_attrs, edge_attrs = self.attribute_names()
        for i, (node_label, node_data) in enumerate(node_values.items()):
            assert type(node_label) == str
            if node_label in node_attrs:
                node_data = [
                    self.G.nodes[node].get(node_label, np.nan)
                    for node in self.G.nodes
                ]
            else:
                assert type(node_data) in [np.ndarray, list]
                assert len(node_data) == n_nodes, f'Invalid input for node values to be displayed: \
                    Provided data has length {len(node_data)} but there are {n_nodes} nodes in the graph.'
                field_values = node_values

            node_values_text = "\n".join(
                f"{v:.16g}"
                for v in node_data
            )

            if i == 0:
                xdmf += """
      <!-- =====================================================
           Scalar value associated with each node
           ===================================================== -->
"""
                    
            xdmf += f"""

      <Attribute
          Name="{node_label}"
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
"""

        for i, (edge_label, edge_data) in enumerate(edge_values.items()):
            assert type(edge_label) == str
            if edge_label in edge_attrs:
                field_values = [
                    self.G.edges[edge].get(edge_data, np.nan)
                    for edge in self.G.edges
                ]
            else:
                assert type(edge_data) in [np.ndarray, list]
                assert len(edge_data) == n_edges, f'Invalid input for edge values to be displayed: \
                    Provided data has length {len(edge_data)} but there are {n_edges} edges in the graph.'
                field_values = edge_values
            
            edge_values_text = "\n".join(
                f"{v:.16g}"
                for v in edge_data
            )

            if i == 0:
                xdmf += """
      <!-- =====================================================
           Scalar value associated with each node
           ===================================================== -->
"""
                 
            xdmf += f"""

      <Attribute
          Name="{edge_label}"
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
"""

        xdmf += """
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
        node_attr: list[list[str]],
        edge_attr: list[str],
        export_stats: bool = False,
    ) -> None:
        nodes = list(self.G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Node features
        nodes_by_type = defaultdict(list)
        node_types = np.zeros(self.G.number_of_nodes(), dtype=int)
        for n, attrs in self.G.nodes(data=True):
            nodes_by_type[attrs.get("type")].append(n)
            node_types[node_to_idx[n]] = attrs.get("type")

        assert len(node_attr) == len(nodes_by_type.items())

        X = dict()
        for t in range(len(nodes_by_type)):
            X[t] = []
            nodes_t = nodes_by_type[t]
            for n in nodes_t:

                features = np.concatenate([
                    np.atleast_1d(self.G.nodes[n].get(f, 0)).ravel()
                    for f in node_attr[t]
                ])
                X[t].extend([features])

        # Edges, both directions
        edges = []
        E = []

        for u, v, data in self.G.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            features = np.concatenate([
                np.atleast_1d(data.get(f, 0)).ravel()
                for f in edge_attr
            ])

            edges.extend([(i, j)])
            E.extend([features])

        edge_index = np.asarray(edges, dtype=np.int64).T
        E = np.asarray(E, dtype=np.float32)

        with h5py.File(filename.with_suffix('.h5'), "a") as h5_file:
            compression_opts = 6

            grp = h5_file.require_group(grp)

            grp["edge_index"] = edge_index
            grp["node_types"] = node_types
            for t in range(len(nodes_by_type)):
                dset = grp.create_dataset(
                    f'x_{t}',
                    data=X[t],
                    dtype='f8',
                    compression='gzip',
                    compression_opts=compression_opts
                )
                dset.attrs['n_nodes_type'] = len(nodes_by_type[t])
                dset.attrs['order'] = str(node_attr[t])

            dset = grp.create_dataset(
                'edge_attr',
                data=E,
                dtype='f8',
                compression='gzip',
                compression_opts=compression_opts
            )
            dset.attrs['n_edges'] = self.G.number_of_edges()
            dset.attrs['order'] = str(edge_attr)

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

        # ---------------------------------------------------------------
        # Enforce periodicity
        # ---------------------------------------------------------------

        if self.periodicity:
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
            "0":  {"type": 1, "position": 3},
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
                    area=tessellation._get_facet_surface(facet_id)
                    )
                self.G.add_edge(
                    _grain_j, _facet_i,
                    distance=-dist,
                    length=np.linalg.norm(dist),
                    area=tessellation._get_facet_surface(facet_id)
                    )

        # Connect facets that share an edge
        # for edge_id, facets in tessellation.edge_facets.items():
        #     _facet_i = _fname(facets.pop(), shift=True)
        #     _facet_j = _fname(facets.pop(), shift=True)
        #     # Identify the vertices forming the edge
        #     (_vertex_k, _vertex_l) = tessellation.edges[edge_id]
        #     _edge_pos = (np.array(tessellation.vertices[_vertex_k]) + np.array(tessellation.vertices[_vertex_l])) / 2
        #     self.G.add_edge(
        #         _facet_i, _facet_j,
        #         # Construct lenght via edge separating the facets
        #         distance=np.array(self.G.nodes[_facet_j]["position"]) -
        #             np.array(self.G.nodes[_facet_i]["position"]),
        #         length=np.linalg.norm(np.array(self.G.nodes[_facet_j]["position"]) - _edge_pos) +
        #             np.linalg.norm(np.array(self.G.nodes[_facet_i]["position"]) - _edge_pos),
        #         area=tessellation._get_edge_length(edge_id) * 0.005
        #     )
        #     self.G.add_edge(
        #         _facet_j, _facet_i,
        #         # Construct length via edge separating the facets
        #         distance=np.array(self.G.nodes[_facet_i]["position"]) -
        #             np.array(self.G.nodes[_facet_j]["position"]),
        #         length=np.linalg.norm(np.array(self.G.nodes[_facet_j]["position"]) - _edge_pos) +
        #             np.linalg.norm(np.array(self.G.nodes[_facet_i]["position"]) - _edge_pos),
        #         area=tessellation._get_edge_length(edge_id) * 0.005
        #     )

        # Connect facets that share two vertices (ALTERNATIVE)
        for facet_i, data_i in tessellation.facets.items():
            vertices_i = data_i["vertices"]
            for facet_j, data_j in tessellation.facets.items():
                if facet_i != facet_j:
                    vertices_j = data_j["vertices"]
                    shared_vertices = np.intersect1d(vertices_i, vertices_j)
                    if len(shared_vertices) == 2:
                        v_1_pos = np.array(tessellation.vertices[shared_vertices[0]])
                        v_2_pos = np.array(tessellation.vertices[shared_vertices[1]])
                        _facet_i = _fname(facet_i, shift=True)
                        _facet_j = _fname(facet_j, shift=True)
                        self.G.add_edge(
                            _facet_i, _facet_j,
                            distance=np.array(self.G.nodes[_facet_j]["position"]) -
                                np.array(self.G.nodes[_facet_i]["position"]),
                            length=np.linalg.norm(np.array(self.G.nodes[_facet_j]["position"]) - (v_1_pos+v_2_pos)/2) +
                                np.linalg.norm(np.array(self.G.nodes[_facet_i]["position"]) - (v_1_pos+v_2_pos)/2),
                            area=np.linalg.norm(v_1_pos - v_2_pos) * 0.005
                        )
                        self.G.add_edge(
                            _facet_j, _facet_i,
                            distance=np.array(self.G.nodes[_facet_i]["position"]) -
                                np.array(self.G.nodes[_facet_j]["position"]),
                            length=np.linalg.norm(np.array(self.G.nodes[_facet_j]["position"]) - (v_1_pos+v_2_pos)/2) +
                                np.linalg.norm(np.array(self.G.nodes[_facet_i]["position"]) - (v_1_pos+v_2_pos)/2),
                            area=np.linalg.norm(v_1_pos - v_2_pos) * 0.005
                        )

        # Account for periodic boundary by merging corresponding facet nodes to primary instance
        mapping = {}
        for secondary_node_idx, data in tessellation.periodicity["facets"].items():
            # Make sure that nodes to be merged do not have common edges (might alter attributes)
            assert len([x for x in list(self.G.neighbors(_fname(data["primary"], shift=True))) 
                        if x in list(self.G.neighbors(_fname(secondary_node_idx, shift=True)))]) == 0
            mapping[_fname(secondary_node_idx, shift=True)] = _fname(data["primary"], shift=True)
        nx.relabel_nodes(self.G, mapping, copy=False)

        self.node_schemas = {
            "0":  {"type": 1, "position": 3},
            "1":  {"type": 1, "position": 3}
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
              "type": 0})
            for grain_id, grain_data in tessellation.seeds.items()
        )
        # Add one node per facet (compute centroid for position)
        self.G.add_nodes_from(
            (_fname(facet_id, shift=True), 
             {"position": np.mean([tessellation.vertices[vertex_id] for vertex_id in facet['vertices']], axis=0),
              "type": 1})
            for facet_id, facet in tessellation.facets.items()
        )
        # Add one node per vertex (position given in tessellation)
        self.G.add_nodes_from(
            (_vname(vertex_id, shift=True), 
             {"position": vertex_data, 
              "type": 2})
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

        # ---------------------------------------------------------------
        # Enforce periodicity
        # ---------------------------------------------------------------

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

        if self.periodicity:
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
            "0":  {"type": 1, "position": 3},
            "1":  {"type": 1, "position": 3},
            "2":  {"type": 1, "position": 3}
        }

    def _add_node_features(
        self,
    ) -> None:
        ...

    def _add_edge_features(
        self,
    ) -> None:
        ...
