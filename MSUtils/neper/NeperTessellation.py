from collections import defaultdict

def _next_line(i, lines):
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i

def _section(name, lines):
    for i, line in enumerate(lines):
        if line.strip() == name:
            return i
    raise ValueError(f"section {name} not found")


class NeperTessellation():
    def __init__(
        self,
        filename,
    ):
        data = self._parse_neper_tess(filename)

        self.seeds = data["seeds"]
        self.seed_data = data["seed_data"]
        self.vertices = data["vertices"]
        self.edges = data["edges"]
        self.faces = data["faces"]
        self.polyhedra = data["polyhedra"]

        self.edge_faces = data["edge_faces"]
        self.face_grains = data["face_grains"]
        self.edge_neighbors = data["edge_neighbors"]

        self.periodicity = data["periodicity"]


    def _parse_neper_tess(self, filename):

        with open(filename) as f:
            lines = f.readlines()

        # ---------------------------------------------------------------
        # SEEDS
        #
        # Find *seed within **cell.
        # The number of seed records is the number of cells.
        # ---------------------------------------------------------------

        cell = _section("**cell", lines)
        seed = next(
            i for i in range(cell + 1, len(lines))
            if lines[i].strip() == "*seed"
        )

        i = seed + 1

        # Number of cells is given at the beginning of **cell
        i = _next_line(cell + 1, lines)
        n_grains = int(lines[i])
        
        i = seed + 1

        seeds = {}
        seed_data = {}

        for _ in range(n_grains):
            i = _next_line(i, lines)
            f = lines[i].split()

            gid = int(f[0])
            xyz = tuple(map(float, f[1:4]))
            weight = float(f[4]) if len(f) > 4 else None

            seeds[gid] = xyz
            seed_data[gid] = {
                "position": xyz,
                "weight": weight,
            }

            i += 1

        # ---------------------------------------------------------------
        # VERTICES
        #
        # vertex_id x y z state
        # ---------------------------------------------------------------

        i = _next_line(_section("**vertex", lines) + 1, lines)
        n = int(lines[i])
        i += 1

        vertices = {}

        for _ in range(n):
            i = _next_line(i, lines)
            f = lines[i].split()

            vertices[int(f[0])] = tuple(map(float, f[1:4]))
            i += 1

        # ---------------------------------------------------------------
        # EDGES
        #
        # edge_id vertex_1 vertex_2 state
        # ---------------------------------------------------------------

        i = _next_line(_section("**edge", lines) + 1, lines)
        n = int(lines[i])
        i += 1

        edges = {}

        for _ in range(n):
            i = _next_line(i, lines)
            f = lines[i].split()

            edges[int(f[0])] = (int(f[1]), int(f[2]))
            i += 1

        # ---------------------------------------------------------------
        # FACES
        #
        # Each face has four lines:
        #   face_id n_vertices v1 ... vn
        #   n_edges e1 ... en
        #   d a b c
        #   state interpolation x y z
        # ---------------------------------------------------------------

        i = _next_line(_section("**face", lines) + 1, lines)
        n = int(lines[i])
        i += 1

        faces = {}

        for _ in range(n):

            # Face ID + vertices
            i = _next_line(i, lines)
            f = lines[i].split()

            fid = int(f[0])
            nv = int(f[1])
            vertex_ids = list(map(int, f[2:2 + nv]))
            i += 1

            # Edges
            i = _next_line(i, lines)
            f = lines[i].split()

            ne = int(f[0])
            edge_ids = list(map(int, f[1:1 + ne]))
            i += 1

            # Plane equation
            i = _next_line(i, lines)
            equation = tuple(map(float, lines[i].split()[:4]))
            i += 1

            # State / interpolation / point
            i = _next_line(i, lines)
            f = lines[i].split()

            state = int(f[0])
            interpolation = int(f[1])
            point = tuple(map(float, f[2:5]))
            i += 1

            faces[fid] = {
                "vertices": vertex_ids,
                "edges": edge_ids,
                "equation": equation,
                "state": state,
                "interpolation": interpolation,
                "point": point,
            }

        # ---------------------------------------------------------------
        # POLYHEDRA / GRAINS
        #
        # Each grain/polyhedron:
        #     polyhedron_id n_faces f1 f2 ...
        # Face IDs are signed.
        # ---------------------------------------------------------------

        i = _next_line(_section("**polyhedron", lines) + 1, lines)
        n = int(lines[i])
        i += 1

        polyhedra = {}

        for _ in range(n):
            i = _next_line(i, lines)
            f = lines[i].split()

            gid = int(f[0])
            nf = int(f[1])

            polyhedra[gid] = list(map(int, f[2:2 + nf]))
            i += 1

        # ---------------------------------------------------------------
        # Derived connectivity
        # ---------------------------------------------------------------

        # edge -> faces
        edge_faces = defaultdict(set)

        for fid, face in faces.items():
            for eid in face["edges"]:
                edge_faces[abs(eid)].add(fid)

        # face -> grains
        face_grains = defaultdict(set)

        for gid, face_ids in polyhedra.items():
            for fid in face_ids:
                face_grains[abs(fid)].add(gid)

        # edge -> grains
        edge_neighbors = {
            eid: {
                gid
                for fid in edge_faces[eid]
                for gid in face_grains[fid]
            }
            for eid in edges
        }

        # ---------------------------------------------------------------
        # PERIODICITY
        #
        # **periodicity
        #
        # *general
        #   per_x per_y per_z
        #   dist_x dist_y dist_z
        #
        # *vertex
        #   n
        #   secondary primary shift_x shift_y shift_z
        #
        # *edge
        #   n
        #   secondary primary shift_x shift_y shift_z orientation
        #
        # *face
        #   n
        #   secondary primary shift_x shift_y shift_z orientation
        # ---------------------------------------------------------------

        periodic = {
            "directions": (0, 0, 0),
            "distances": (0.0, 0.0, 0.0),
            "vertices": {},
            "edges": {},
            "faces": {},
        }

        # Find periodicity section, if present
        periodic_start = None

        for j, line in enumerate(lines):
            if line.strip() == "**periodicity":
                periodic_start = j
                break

        if periodic_start is not None:

            # -----------------------------------------------------------
            # *general
            # -----------------------------------------------------------

            i = _next_line(periodic_start + 1, lines)

            while lines[i].strip() != "*general":
                i += 1

            i = _next_line(i + 1, lines)

            periodic["directions"] = tuple(
                map(int, lines[i].split()[:3])
            )

            i = _next_line(i + 1, lines)

            periodic["distances"] = tuple(
                map(float, lines[i].split()[:3])
            )

            # -----------------------------------------------------------
            # Parse a periodic subsection
            # -----------------------------------------------------------

            def parse_periodic_section(name, n_values):
                for j in range(periodic_start + 1, len(lines)):
                    if lines[j].strip() == name:
                        i = _next_line(j + 1, lines)
                        n = int(lines[i])
                        i += 1

                        result = {}

                        for _ in range(n):
                            i = _next_line(i, lines)
                            f = lines[i].split()

                            secondary = int(f[0])
                            primary = int(f[1])
                            shift = tuple(map(int, f[2:5]))

                            data = {
                                "primary": primary,
                                "shift": shift,
                            }

                            if n_values == 6:
                                data["orientation"] = int(f[5])

                            result[secondary] = data
                            i += 1

                        return result

                return {}

            # -----------------------------------------------------------
            # Periodic vertices
            # -----------------------------------------------------------

            periodic["vertices"] = parse_periodic_section(
                "*vertex", 5
            )

            # -----------------------------------------------------------
            # Periodic edges
            # -----------------------------------------------------------

            periodic["edges"] = parse_periodic_section(
                "*edge", 6
            )

            # -----------------------------------------------------------
            # Periodic faces
            # -----------------------------------------------------------

            periodic["faces"] = parse_periodic_section(
                "*face", 6
            )

        return {
            "seeds": seeds,
            "seed_data": seed_data,
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "polyhedra": polyhedra,
            "edge_faces": dict(edge_faces),
            "face_grains": dict(face_grains),
            "edge_neighbors": edge_neighbors,
            "periodicity": periodic
        }
