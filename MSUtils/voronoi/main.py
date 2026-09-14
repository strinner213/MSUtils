from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.voronoi.VoronoiGBErosion import PeriodicVoronoiImageErosion
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.voronoi.VoronoiOrientations import VoronoiOrientations
from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation


def main():
    num_crystals = 27
    L = [1, 1, 1]
    Nx, Ny, Nz = 128, 128, 128
    permute_order = "zyx"

    # Generate Voronoi seeds and tessellation
    SeedInfo = VoronoiSeeds(num_crystals, L, "sobol", BitGeneratorSeed=42)
    voroTess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)

    # Generate orientations for grains
    voroOri = VoronoiOrientations(SeedInfo.num_crystals, texture='random', symmetry='mmm', sampling='uniform', descriptor='quaternion:active', seed=42)
    #voroOri = VoronoiOrientations(SeedInfo.num_crystals, texture='uniform', descriptor='quaternion:active', ori_quat=[-0.5, 0.5, -0.5, -0.5])
    
    # Test for average global preferred directions
    avg_orientation = voroOri.characterize_orientations(10., 5., 1.)
    print(avg_orientation)
    
    # Generate Voronoi image
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], voroTess.seeds, L)

    # Export Voronoi structure without explicit grain boundary representation
    grp_name = "/dset_0"
    IMG = MicrostructureImage(image=voroImg.image, L=L)
    IMG.write("data/voroImg.h5", f"{grp_name}/image", order=permute_order)
    # Add orientations for grains
    voroOri.add_to_h5(IMG.h5_filename, grp_name, "image", voroImg.image, order=permute_order, save_orientations=True)
    write_xdmf(
        "data/voroImg.h5",
        "data/voroImg.xdmf",
        microstructure_length=[1,1,1]
    )

    # Generate Voronoi image with grain boundaries of a specific thickness
    interface_thickness = (1.0 / 128) * 4
    voroErodedImg = PeriodicVoronoiImageErosion(
        voroImg, voroTess, interface_thickness=interface_thickness
    )

    # Export Voronoi structure with explicit grain boundary representation
    voroErodedImg.write_h5(
        "data/voroImg_eroded.h5", grp_name, order=permute_order, save_normals=True
    )
    # Add orientations for grains
    voroOri.add_to_h5("data/voroImg_eroded.h5", grp_name, "eroded_image", voroErodedImg.eroded_image, permute_order, save_orientations=True)
    write_xdmf(
        "data/voroImg_eroded.h5",
        "data/voroImg_eroded.xdmf",
        microstructure_length=[1, 1, 1],
    )

    # Calculate and print volume fraction of all grain boundary (all tags >= num_crystals)
    msimage = MicrostructureImage(image=voroErodedImg.eroded_image, L=L)
    gb_volume_fraction = 0
    for phase, fraction in msimage.volume_fractions.items():
        if phase >= num_crystals:
            gb_volume_fraction += fraction

    gb_volume_fraction_percent = gb_volume_fraction * 100
    print(f"Volume fraction of all grain boundaries: {gb_volume_fraction_percent:.8f}%")
    print(f"Interface thickness: {interface_thickness:.10f}")


if __name__ == "__main__":
    main()
