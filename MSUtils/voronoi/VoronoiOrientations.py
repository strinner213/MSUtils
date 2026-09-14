from pathlib import Path
from typing import Self
import numpy  as np
import numpy.typing as npt
import h5py
import json
from scipy.spatial.transform import Rotation
import subprocess


class VoronoiOrientations:
    def __init__(
        self,
        num_crystals: int,
        texture: str='random',
        descriptor: str='quaternion:active',
        **kwargs
    ) -> Self:
        """
        Class to handle grain orientations.

        Generate orientations for grains (crystals) with the option to use Neper sampling functionalities.

        Args:
            num_crystals (int):  Number of seed points (or crystals).
            texture (str, optional): Intended texture of the polycrystal. Defaults to 'random'. 
            descriptor (str, optional): Descriptor and convention for storing orienations. Defaults to 'quaternion:active'.
            kwargs: Additional options. The supported options depend on 'texture':

                **For `texture == "uniform"` (same orientation for all grains):**
                   - `ori_quat` (list[float]) -- Global orienation in quaternion representation using wxyz-convention. 

                **For `texture == "random"` (sampling via Neper):**
                   - `symmetry` (str) -- Symmetry class of the polycrystal (`cubic` for isotropic, `mmm` for orthotropic)
                   - `sampling` (str) -- Sampling strategy. Defaults to `uniform`.
                   - `seed` (int) -- Random generator seed for reproducability of the sampling.

        Returns:
            Self: Voronoi Orientations Object

        Notes:
            - texture can be one of the following: "random", "uniform".
        """
        self.orientations = None

        self.num_crystals = num_crystals
        self.texture = texture
        self.descriptor = descriptor

        match self.texture:
            case 'uniform': 
                self._generate_uniform_orientations(**kwargs)
            case 'random':
                return self._generate_orientations_with_neper(**kwargs)
            case _:
                raise ValueError(f"Unknown texture type: {texture}")

    def _generate_uniform_orientations(self, ori_quat: list[float]) -> None:
        self.orientations = np.tile(ori_quat, (self.num_crystals, 1))

    def _generate_orientations_with_neper(self, symmetry: str='mmm', sampling: str='uniform', seed: int=42) -> None:
        _path_temp = Path("data/temp_orientations.ori")
        _path_temp.parent.mkdir(parents=True, exist_ok=True)

        # Call Neper to sample orientations
        subprocess.run(
                f'neper -T -o {_path_temp.with_suffix('')} -n {self.num_crystals} -id {seed} -periodicity all -crysym {symmetry} -ori {self.texture} -orisampling {sampling} -format ori -oridescriptor {self.descriptor} -oriformat plain',
                shell=True,
            )

        # Read quaternion from file (format is w | x | y | z)
        self.orientations = np.loadtxt(_path_temp)

        # Delete temporary Neper file
        _path_temp.unlink()

    def characterize_orientations(self, sample_a:float = 10., sample_b:float = 5., sample_c:float = 1.) -> npt.Arraylike:
        """
        Analyze sampled orientations.

        Make sure that rotation for each grain is valid (trace is preserved). Average global orientation is computed
        to check for global preferred directions.

        Args:
            sample_a (float, optional): First component of sample material tensor. Defaults to 10.
            sample_b (float, optional): Second component of sample material tensor. Defaults to 5.
            sample_c (float, optional): Third component of sample material tensor. Defaults to 1.

        Returns:
            np.ndarray: Average global orientation.
        """
        avg_orientation = np.zeros((3,3))
        for i in range(self.num_crystals):
            r = Rotation.from_quat(self.orientations[i,[1,2,3,0]]).as_matrix()
            ori_i = r @ np.diag([sample_a, sample_b, sample_c]) @ r.T
            assert np.isclose(np.trace(ori_i), sample_a + sample_b + sample_c)
            avg_orientation += ori_i

        return avg_orientation / self.num_crystals

    def add_to_h5(
        self,
        filepath: Path,
        grp_name: str,
        dset_name: str,
        image: npt.ArrayLike,
        order: str = "zyx",
        save_orientations: bool = False,
    ) -> None:
        """
        Export grain orientations to existing h5-file.

        Can be applied to both `PeriodicVoronoiImage`and `PeriodicVoronoiImageErosion`.

        Args:
            filepath (Path): Path to existing h5 file.
            grp_name (str): Name of existing h5 group.
            dset_name (str): Name of existing dataset.
            image (np.ndarray): Corresponding voxelized image.
            order (str, optional): Either one of 'xyz' or 'zyx'. Defaults to "zyx".
            save_orientations (bool, optional): Whether to save the field of principal directions. Defaults to False.

        Returns:
            None.
        """
        with h5py.File(filepath, "a") as h5_file:
            grp = h5_file.require_group(grp_name)
            compression_opts = 6

            # Define dtype for the orientation metadata of crystals
            CrystalVoxelInfo_dtype = np.dtype(
                [
                    ("grain_tag", "i8"),
                    ("orientation_wxyz", "f8", (4,))
                ]
            )
            # Convert grain orientations to structured array
            CrystalVoxelInfo = np.array(
                [
                    (
                        tag,
                        self.orientations[tag]
                    )
                    for tag in range(self.num_crystals)
                ],
                dtype=CrystalVoxelInfo_dtype,
            )

            # Reference to image in .h5 file
            image_dataset = grp[f'{dset_name}']

            # Create CrystalVoxelInfo attribute with grain_tag and orientation as key-value pairs
            crystal_voxel_info = {}
            for tag in range(self.num_crystals):
                crystal_voxel_info[str(tag)] = {
                    "grain_tag": int(tag),
                    "orientation_wxyz": self.orientations[tag].tolist(),
                }
            # Store as string attribute (JSON format)
            image_dataset.attrs["CrystalVoxelInfo"] = json.dumps(crystal_voxel_info)

            # Save orientation metadata to .h5 file
            if "CrystalVoxelInfo" in grp:
                del grp["CrystalVoxelInfo"]
                print("Overwriting existing 'CrystalVoxelInfo' dataset.")
            grp.create_dataset(
                "CrystalVoxelInfo",
                data=CrystalVoxelInfo,
                compression="gzip",
                compression_opts=compression_opts,
            )

            # Only save orientation vectors if explicitly requested
            if save_orientations:
                # Create a new field for orientation vectors, taking shape from the original image
                Nx, Ny, Nz = image.shape
                orientation_a_field = np.zeros((Nx, Ny, Nz, 3))
                orientation_b_field = np.zeros((Nx, Ny, Nz, 3))
                orientation_c_field = np.zeros((Nx, Ny, Nz, 3))
                for i, j, k in np.ndindex(image.shape):
                    mat_index = image[i, j, k]
                    if mat_index < self.num_crystals:
                        # Translate quaternion representation to matrix
                        # Note: scipy uses xyzw convention
                        rot_matrix = Rotation.from_quat(self.orientations[mat_index,[1,2,3,0]]).as_matrix()
                        orientation_a_field[i, j, k] = rot_matrix[:,0]
                        orientation_b_field[i, j, k] = rot_matrix[:,1]
                        orientation_c_field[i, j, k] = rot_matrix[:,2]

                # Apply permutation to orientation vector fields if needed
                if order == "xyz":
                    permuted_orientation_a_field = orientation_a_field
                    permuted_orientation_b_field = orientation_b_field
                    permuted_orientation_c_field = orientation_c_field
                elif order == "zyx":
                    permuted_orientation_a_field = orientation_a_field.transpose(2, 1, 0, 3)
                    permuted_orientation_b_field = orientation_b_field.transpose(2, 1, 0, 3)
                    permuted_orientation_c_field = orientation_c_field.transpose(2, 1, 0, 3)

                # Save orientation vectors to .h5 file
                if "orientation_a" in grp:
                    del grp["orientation_a"]
                    print("Overwriting existing 'orientation_a' dataset.")
                orientation_a_dataset = grp.create_dataset(
                    "orientation_a",
                    data=permuted_orientation_a_field,
                    dtype="f8",
                    compression="gzip",
                    compression_opts=compression_opts,
                )
                orientation_a_dataset.attrs["permute_order"] = order

                if "orientation_b" in grp:
                    del grp["orientation_b"]
                    print("Overwriting existing 'orientation_b' dataset.")
                orientation_b_dataset = grp.create_dataset(
                    "orientation_b",
                    data=permuted_orientation_b_field,
                    dtype="f8",
                    compression="gzip",
                    compression_opts=compression_opts,
                )
                orientation_b_dataset.attrs["permute_order"] = order

                if "orientation_c" in grp:
                    del grp["orientation_c"]
                    print("Overwriting existing 'orientation_c' dataset.")
                orientation_c_dataset = grp.create_dataset(
                    "orientation_c",
                    data=permuted_orientation_c_field,
                    dtype="f8",
                    compression="gzip",
                    compression_opts=compression_opts,
                )
                orientation_c_dataset.attrs["permute_order"] = order
