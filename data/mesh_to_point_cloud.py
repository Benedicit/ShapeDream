import glob
import os
import time

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

OBJAVERSE = "objaverse"
REDWOOD = "redwood"
GSO = "gso"

working_dir = os.path.dirname(os.path.realpath(__file__))
original_datasets_dirs = {
	OBJAVERSE: "/.objaverse",
	REDWOOD: "/.redwood",
	GSO: "/.gso",
}
dataset_dir = working_dir + "/dataset"


def random_camera(batch_size=1, min_elev=20.0, max_elev=75.0,
				  min_dist=1.5, max_dist=3.5, image_size=512, fov=60.0, device="cpu"):

	# Azimuth in Degree [0,360), Elevation in Degree, Distance as a factor
	azim = torch.rand(batch_size) * 360.0
	elev = min_elev + torch.rand(batch_size) * (max_elev - min_elev)
	dist = min_dist + torch.rand(batch_size) * (max_dist - min_dist)
	R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
	cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov))
	return cameras, int(image_size), int(image_size)

def download_from_objaverse():
	objaverse_dir = working_dir + original_datasets_dirs[OBJAVERSE]
	os.makedirs(objaverse_dir, exist_ok=True)
	os.makedirs(dataset_dir, exist_ok=True)

	annotations = oxl.get_annotations(download_dir=objaverse_dir)
	#TODO: Scale up
	sampled_df = annotations.groupby('source').apply(lambda x: x.sample(1)).reset_index(drop=True)
	oxl.download_objects(download_dir=objaverse_dir, objects=sampled_df)

def download_from_redwood():
	#TODO
	pass
def download_from_gso():
	#TODO
	pass
def download_from_datasets():
	#TODO
	pass

def generate_point_clouds(datasets: list, percent_saved=0.25, show_first_cloud=False):

	models = []
	for d in datasets:
		if d not in original_datasets_dirs:
			continue
		current_dir = working_dir + original_datasets_dirs[d]
		file_patterns = [rf"{current_dir}/{suffix}" for suffix in ("**/*.glb", "**/*.gltf", "**/*.obj", "**/*.ply", "**/*.stl")]
		for pattern in file_patterns:
			models += glob.glob(pattern, recursive=True)

	if not models:
		print("No models found")
		return
	model_path = Path(models[2])

	start = time.time()
	mesh = trimesh.load_mesh(model_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	verts_t = torch.asarray(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)  # (1, V, 3)
	faces_t = torch.asarray(mesh.faces, dtype=torch.int64, device=device).unsqueeze(0)  # (1, F, 3)

	# Norm coordinates
	verts_center = verts_t.mean(dim=1, keepdim=True)
	scale = (verts_t - verts_center).abs().max()
	verts_t = (verts_t - verts_center) / scale
	meshes = Meshes(verts=verts_t, faces=faces_t)

	cameras, H, W = random_camera(batch_size=1, image_size=512)
	raster_settings = RasterizationSettings(
		image_size=H,
		blur_radius=0.0,
		faces_per_pixel=1,
	)

	rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
	fragments = rasterizer(meshes)
	pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H, W)
	bary_coords = fragments.bary_coords[0, ..., 0, :]  # (H, W, 3)

	# If negative, then it was not visible
	valid_mask = pix_to_face >= 0  # Bool (H, W)

	# face_idx: (P,) P = visible pixels
	face_idx = pix_to_face[valid_mask]
	bary = bary_coords[valid_mask]  # (P, 3)

	faces_packed = meshes.faces_packed()  # (F, 3)
	verts_packed = meshes.verts_packed()  # (V, 3)

	face_verts_idx = faces_packed[face_idx]  # (P, 3)

	# Vertex-Coordinates
	v0 = verts_packed[face_verts_idx[:, 0]]  # (P, 3)
	v1 = verts_packed[face_verts_idx[:, 1]]  # (P, 3)
	v2 = verts_packed[face_verts_idx[:, 2]]  # (P, 3)
	print(f"v0.shape = {v0.shape}")
	print(f"bary.shape = {bary.shape}")
	# baryzentrische interpolation, slicing for correct broadcasting: bary[:, 0:1].shape = (P, 1)
	points = bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2  # (P, 3)

	# To simulate noisy real-world data taking up less memory, only take ~25% of the points
	noise_mask = torch.rand(points.shape[0], device=device) < percent_saved

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points[noise_mask])
	o3d.io.write_point_cloud(Path(f"{dataset_dir}/partial_pointcloud.ply"), pcd)

	end = time.time()

	print(f"partial_pointcloud.ply: points = {len(points)})")
	print(f"Time elapsed: {end - start}")
	if show_first_cloud:
		o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
	generate_point_clouds([OBJAVERSE], show_first_cloud=True)