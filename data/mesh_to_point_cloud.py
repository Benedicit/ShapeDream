import glob
import os
import torch
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras,
	RasterizationSettings,
	MeshRasterizer
)

import objaverse.xl as oxl


def random_camera(batch_size=1, min_elev=10.0, max_elev=80.0,
				  min_dist=1.5, max_dist=3.5, image_size=512, fov=60.0):

	# Azimuth in Degree [0,360), Elevation in Degree, Distance as a factor
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	azim = torch.rand(batch_size) * 360.0
	elev = min_elev + torch.rand(batch_size) * (max_elev - min_elev)
	dist = min_dist + torch.rand(batch_size) * (max_dist - min_dist)
	R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
	cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov))
	return cameras, int(image_size), int(image_size)

def download_models(download=True):
	working_dir = os.path.dirname(os.path.realpath(__file__))
	objaverse_dir = working_dir + "/.objaverse"
	dataset_dir = working_dir + "/dataset"
	os.makedirs(objaverse_dir, exist_ok=True)
	os.makedirs(dataset_dir, exist_ok=True)

	annotations = oxl.get_annotations(download_dir=objaverse_dir)
	if download:
		sampled_df = annotations.groupby('source').apply(lambda x: x.sample(1)).reset_index(drop=True)
		oxl.download_objects(download_dir=objaverse_dir, objects=sampled_df)

	#models = [each for each in os.listdir(objaverse_dir) if each.endswith((".glb", ".gltf", ".obj", ".ply", ".stl"))]

	file_patterns = [rf"{objaverse_dir}/{suffix}" for suffix in ("**/*.glb", "**/*.gltf", "**/*.obj", "**/*.ply", "**/*.stl")]
	models = []
	for pattern in file_patterns:
		models += glob.glob(pattern, recursive=True)

	if not models:
		print("No models found in %s" % objaverse_dir)
		return

	model_path = Path(models[2])

	mesh = trimesh.load_mesh(model_path)

	verts = np.asarray(mesh.vertices, dtype=np.float32)
	faces = np.asarray(mesh.faces, dtype=np.int64)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	verts_t = torch.from_numpy(verts).unsqueeze(0).to(device)  # (1, V, 3)
	faces_t = torch.from_numpy(faces).unsqueeze(0).to(device)  # (1, F, 3)

	# Norm coordinates
	verts_center = verts_t.mean(dim=1, keepdim=True)
	scale = (verts_t - verts_center).abs().max()
	verts_t = (verts_t - verts_center) / scale
	meshes = Meshes(verts=verts_t, faces=faces_t)

	cameras, H, W = random_camera(batch_size=1, image_size=512)
	raster_settings = RasterizationSettings(
		image_size=H,
		blur_radius=0.0,
		faces_per_pixel=1,  # nur die vorderste Oberfläche
	)

	rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
	fragments = rasterizer(meshes)  # Fragments enthält zbuf, pix_to_face, bary_coords
	# fragments.zbuf shape: (N, H, W, K)  (K == faces_per_pixel)
	zbuf = fragments.zbuf[0, ..., 0]  # (H, W)

	# Maske der sichtbaren Pixel
	valid_mask = torch.isfinite(zbuf)

	# Erzeuge Pixel-Koordinaten (pixel centers)
	ys = torch.arange(H, device=device, dtype=torch.float32)
	xs = torch.arange(W, device=device, dtype=torch.float32)
	grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

	u = (grid_x + 0.5) / float(W)
	v = (grid_y + 0.5) / float(H)
	x_ndc = 2.0 * u - 1.0
	y_ndc = 1.0 - 2.0 * v

	z_valid = zbuf[valid_mask]  # (P,)
	x_ndc_valid = x_ndc[valid_mask]
	y_ndc_valid = y_ndc[valid_mask]

	# new shape (1, P, 3) for points
	xy_depth = torch.stack([x_ndc_valid, y_ndc_valid, z_valid], dim=1).unsqueeze(0)

	points_world = cameras.unproject_points(xy_depth, world_coordinates=True, from_ndc=True)
	points = points_world[0].cpu().numpy()  # (P, 3)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	o3d.io.write_point_cloud(Path(f"{dataset_dir}/partial_pointcloud.ply"), pcd)
	print(f"partial_pointcloud.ply: points = {len(points)})")
	o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
	download_models(download=False)