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
import itertools

from MSUtils.neper.NeperTessellation import NeperTessellation


def _gname(grain_id: int):
    return f'g_{grain_id-1:05}'

def _vname(vertex_id: int):
    return f'v_{vertex_id-1:05}'

def _fname(face_id: int):
    return f'f_{face_id-1:05}'


class PolycrystalGraph:
    def __init__(
        self,
        tessellation: NeperTessellation,
        node_features: list[str],
        edge_features: list[str],
    ) -> Self:

        self.G = nx.Graph()

        self._construct_topology(tessellation)

        self.node_feature_labels = node_features
        self._add_node_features()

        self.edge_feature_labels = edge_features
        self._add_edge_features()

    def _construct_topoplogy(self, tessellation: NeperTessellation):
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
        print("Graph statisics")
        print("=========================")
        print(f"Nodes:       {self.G.number_of_nodes():,}")
        print(f"Edges:       {self.G.number_of_edges():,}")
        print(f"Density:     {nx.density(self.G):.4f}")
        print(f"Connected:   {nx.is_connected(self.G)}")
        print(f"Components:  {nx.number_connected_components(self.G)}")
        print(f"Avg degree:  {sum(degrees) / len(degrees):.2f}")
        print(f"Max degree:  {max(degrees):,}")
        print(f"Min degree:  {min(degrees):,}\n")

    def visualize_3d(
        self,
        filepath: Path,
        node_feature: str | None = None,
    ):
        """Visualize a 3D NetworkX graph as a rotating transparent GIF.

        Each node of the NetworkX graph must have a 3D ``"position"`` attribute.

        Parameters
        ----------
        filepath:
            Output path. The GIF is written using the same path with a ``.gif`` suffix.
        node_feature:
            Optional node attribute containing a scalar value used to color the nodes.
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
            for u, v in self.G.edges
        ]

        # --------------------------------------------------
        # Node colors
        # --------------------------------------------------

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

        if node_feature is not None:
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
            features = [data.get(f, 0) for f in edge_features]

            edges.extend([(i, j), (j, i)])
            E.extend([features, features])

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
    def __init__(
        self,
        tessellation: NeperTessellation,
        node_features: list[str],
        edge_features: list[str],
    ) -> Self:
        super().__init__(tessellation, node_features, edge_features)

    def _construct_topology(
        self, 
        tessellation: NeperTessellation
    ):

        # ---------------------------------------------------------------
        # Add nodes
        # ---------------------------------------------------------------

        # Add one node per grain (position given by seed in tessellation)
        self.G.add_nodes_from(
            (_gname(grain_id), 
             {"position": grain_data, 
              "type": 0,
              "boundary": 0})
            for grain_id, grain_data in tessellation.seeds.items()
        )

        # ---------------------------------------------------------------
        # Add edges
        # ---------------------------------------------------------------

        # Connect grains that share a face
        for face_id, grains in tessellation.face_grains.items():
            if len(grains) == 2:
                self.G.add_edge(_gname(grains.pop()), _gname(grains.pop()))

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
    def __init__(
        self,
        tessellation: NeperTessellation,
        node_features: list[str],
        edge_features: list[str],
    ) -> Self:
        super().__init__(tessellation, node_features, edge_features)

    def _construct_topology(
        self,
        tessellation: NeperTessellation
    ) -> None:

        # ---------------------------------------------------------------
        # Add nodes
        # ---------------------------------------------------------------

        # Add one node per grain (position given by seed in tessellation)
        self.G.add_nodes_from(
            (_gname(grain_id), 
                {"position": grain_data, 
                 "type": 0,
                 "boundary": 0})
            for grain_id, grain_data in tessellation.seeds.items()
        )
        # Add one node per face (compute centroid for position)
        self.G.add_nodes_from(
            (_fname(face_id), 
                {"position": np.mean([tessellation.vertices[vertex_id] for vertex_id in face['vertices']], axis=0),
                 "type": 1,
                 "boundary": 0})
            for face_id, face in tessellation.faces.items()
        )

        # ---------------------------------------------------------------
        # Add edges
        # ---------------------------------------------------------------

        # Connect faces to grains they separate
        for face_id, grains in tessellation.face_grains.items():
            for _ in range(len(grains)):
                self.G.add_edge(_fname(face_id), _gname(grains.pop()))

        # Connect faces that share an edge
        for edge_id, faces in tessellation.edge_faces.items():
            self.G.add_edge(_fname(faces.pop()), _fname(faces.pop()))

        # ---------------------------------------------------------------
        # Mark boundary nodes for periodic tessellation
        # ---------------------------------------------------------------

        # Tag periodicity relations of faces
        nx.set_node_attributes(self.G, [np.nan, np.nan, np.nan], "boundary_shift")
        for secondary_node_idx, data in tessellation.periodicity["faces"].items():
            self.G.nodes[_fname(secondary_node_idx)]["boundary"] = 2
            self.G.nodes[_fname(data["primary"])]["boundary"] = 1
            self.G.nodes[_fname(data["primary"])]["boundary_shift"] = data["shift"]

        self.node_schemas = {
            "0":  {"type": 1, "position": 3, "boundary": 1},
            "1":  {"type": 1, "position": 3, "boundary": 1, "boundary_shift": 3}
        }

    def _add_node_features(
        self,
    ) -> None:
        ...

    def _add_edge_features(
        self,
    ) -> None:
        ...

class PolycrystalVertexEnhancedGraph(PolycrystalGraph):
    def __init__(
        self,
        tessellation: NeperTessellation,
        node_features: list[str],
        edge_features: list[str],
    ) -> Self:
        super().__init__(tessellation, node_features, edge_features)

    def _construct_topology(
        self,
        tessellation: NeperTessellation
    ) -> None:
        
        # ---------------------------------------------------------------
        # Add nodes
        # ---------------------------------------------------------------

        # Add one node per grain (position given by seed in tessellation)
        self.G.add_nodes_from(
            (_gname(grain_id), 
             {"position": grain_data, 
              "type": 0,
              "boundary": 0})
            for grain_id, grain_data in tessellation.seeds.items()
        )
        # Add one node per face (compute centroid for position)
        self.G.add_nodes_from(
            (_fname(face_id), 
                {"position": np.mean([tessellation.vertices[vertex_id] for vertex_id in face['vertices']], axis=0),
                "type": 1,
                "boundary": 0})
            for face_id, face in tessellation.faces.items()
        )
        # Add one node per vertex (position given in tessellation)
        self.G.add_nodes_from(
            (_vname(vertex_id), 
             {"position": vertex_data, 
              "type": 2,
              "boundary": 0})
            for vertex_id, vertex_data in tessellation.vertices.items()
        )

        # ---------------------------------------------------------------
        # Add edges
        # ---------------------------------------------------------------

        # Connect faces to grains they separate
        for face_id, grains in tessellation.face_grains.items():
            for _ in range(len(grains)):
                self.G.add_edge(_fname(face_id), _gname(grains.pop()))

        # Connect vertices that form an edge
        for edge_id, (v1, v2) in tessellation.edges.items():
            self.G.add_edge(_vname(v1), _vname(v2))

        # Connect vertices to the centroid of the face they form
        for face_id, face in tessellation.faces.items():
            self.G.add_edges_from([(_fname(face_id), _vname(vertex_id)) for vertex_id in face['vertices']])

        # ---------------------------------------------------------------
        # Mark boundary nodes for periodic tessellation
        # ---------------------------------------------------------------

        # Tag periodicity relations of vertices
        nx.set_node_attributes(self.G, [np.nan, np.nan, np.nan], "boundary_shift")
        for secondary_node_idx, data in tessellation.periodicity["vertices"].items():
            self.G.nodes[_vname(secondary_node_idx)]["boundary"] = 2
            self.G.nodes[_vname(data["primary"])]["boundary"] = 1
            self.G.nodes[_vname(data["primary"])]["boundary_shift"] = data["shift"]

        # Tag periodicity relations of faces
        nx.set_node_attributes(self.G, [np.nan, np.nan, np.nan], "boundary_shift")
        for secondary_node_idx, data in tessellation.periodicity["faces"].items():
            self.G.nodes[_fname(secondary_node_idx)]["boundary"] = 2
            self.G.nodes[_fname(data["primary"])]["boundary"] = 1
            self.G.nodes[_fname(data["primary"])]["boundary_shift"] = data["shift"]

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

        for feature in self.edge_feature_labels:
            match feature:
                case 'distance':
                    distances = {
                        (u, v): np.linalg.norm(
                            np.asarray(self.G.nodes[u]["position"]) - np.asarray(self.G.nodes[v]["position"])
                        )
                        for u, v in self.G.edges
                    }

                    nx.set_edge_attributes(self.G, distances, "distance")

