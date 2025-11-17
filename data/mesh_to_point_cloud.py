import glob
import os
import time
import csv
import point_cloud_utils as pcu
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

from tqdm import tqdm

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
RECURSIVE_FILE_PATHS = ("**/*.glb", "**/*.gltf", "**/*.obj", "**/*.ply", "**/*.stl")

def get_models_from_datasets(datasets: list):
	models = []
	labels = dict()
	for d in datasets:
		if d not in original_datasets_dirs:
			continue
		if d == GSO:
			with open(f"{working_dir}/{GSO}_labels.csv", encoding="utf-8") as f:
				temp = {row["filename"]: row["label"] for row in csv.DictReader(f)}
				labels.update(temp)

		current_dir = working_dir + original_datasets_dirs[d]
		file_patterns = [rf"{current_dir}/{suffix}" for suffix in
						 RECURSIVE_FILE_PATHS]
		for pattern in file_patterns:
			models += glob.glob(pattern, recursive=True)
	labels = {m: labels[os.path.basename(m)] for m in models if os.path.basename(m) in labels}

	return labels

def random_camera(batch_size=1, min_elev=20.0, max_elev=75.0,
				  min_dist=1.5, max_dist=3.5, image_size=512, fov=60.0, device="cpu"):

	# Azimuth in Degree [0,360), Elevation in Degree, Distance as a factor
	azim = torch.rand(batch_size) * 360.0
	elev = min_elev + torch.rand(batch_size) * (max_elev - min_elev)
	dist = min_dist + torch.rand(batch_size) * (max_dist - min_dist)
	R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
	cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov))
	return cameras, int(image_size), int(image_size)

def point_cloud_from_camera_view(models: dict, percent_saved=0.50, show_first_cloud=False):


	if not models:
		print("No models found")
		return
	# TODO: BATCH
	model_path = Path(models[2])

	start = time.time()
	mesh = trimesh.load_mesh(model_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	verts = torch.asarray(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)  # (1, V, 3)
	faces = torch.asarray(mesh.faces, dtype=torch.int64, device=device).unsqueeze(0)  # (1, F, 3)

	# Norm coordinates
	verts_center = verts.mean(dim=1, keepdim=True)
	scale = (verts - verts_center).abs().max()
	verts = (verts - verts_center) / scale
	meshes = Meshes(verts=verts, faces=faces)

	cameras, H, W = random_camera(batch_size=1, image_size=512, device=device)
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
	# barycentric interpolation, slicing for correct broadcasting: bary[:, 0:1].shape = (P, 1)
	points = bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2  # (P, 3)

	# To simulate noisy real-world data taking up less memory, only take ~25% of the points
	noise_mask = torch.rand(points.shape[0], device=device) < percent_saved
	pcu.save_mesh_v(f"{dataset_dir}/partial_pointcloud.ply", points[noise_mask].cpu().numpy())

	end = time.time()

	print(f"partial_pointcloud.ply: points = {len(points)})")
	print(f"Time elapsed: {end - start}")
	if show_first_cloud:
		o3d.visualization.draw_geometries([points])


def point_cloud_from_mesh(models: dict, percent_saved=0.90, n_points=200_000, points_at_once=1_000_000, show_first_cloud=False, normalise=True,
						  device="cpu", seed=42):
	model_path = Path(models[2])

	gen = torch.Generator(device=device).manual_seed(seed)
	mesh = trimesh.load_mesh(model_path)

	start = time.time()
	verts = torch.asarray(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)  # (1, V, 3)
	faces = torch.asarray(mesh.faces, dtype=torch.int64, device=device).unsqueeze(0)  # (1, F, 3)

	if normalise:
		verts_center = verts.mean(dim=1, keepdim=True)
		scale = (verts - verts_center).abs().max()
		verts = (verts - verts_center) / scale

	meshes = Meshes(verts=verts, faces=faces)
	faces_packed = meshes.faces_packed()  # (F, 3)
	verts_packed = meshes.verts_packed()  # (V, 3)

	# Vertex-Coordinates
	v0 = verts_packed[faces_packed[:, 0]]  # (P, 3)
	v1 = verts_packed[faces_packed[:, 1]]  # (P, 3)
	v2 = verts_packed[faces_packed[:, 2]]  # (P, 3)

	cross = torch.cross(v1 - v0, v2 - v0, dim=1)
	face_areas = 0.5 * torch.norm(cross, dim=1)  # (F,)
	face_probs = face_areas / face_areas.sum()

	points_sampled = []
	remaining_points = n_points
	while remaining_points > 0:
		point_batch = min(points_at_once, remaining_points)
		face_idx = torch.multinomial(face_probs, num_samples=point_batch, replacement=True, generator=gen)

		# barycentric Sampling
		u = torch.rand(point_batch, device=device, generator=gen)
		v = torch.rand(point_batch, device=device, generator=gen)
		r1 = torch.sqrt(u)
		r2 = v
		w0 = (1.0 - r1).unsqueeze(1)  # (b,1)
		w1 = (r1 * (1.0 - r2)).unsqueeze(1)
		w2 = (r1 * r2).unsqueeze(1)

		chosen_faces = faces_packed[face_idx]  # (b,3)
		chosen_tri = verts_packed[chosen_faces]  # (b,3,3)
		p0 = chosen_tri[:, 0, :]
		p1 = chosen_tri[:, 1, :]
		p2 = chosen_tri[:, 2, :]
		pts = w0 * p0 + w1 * p1 + w2 * p2  # (b,3)
		points_sampled.append(pts)
		remaining_points -= point_batch

	points_sampled = torch.cat(points_sampled, dim=0)

	noise_mask = torch.rand(points_sampled.shape[0], device=device) < percent_saved

	pcu.save_mesh_v(f"{dataset_dir}/partial_pointcloud.ply", points_sampled[noise_mask].cpu().numpy())

	end = time.time()

	if show_first_cloud:
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points_sampled[noise_mask])
		o3d.visualization.draw_geometries([pcd])
	print(f"Time elapsed: {end - start}")

def pcu_based_generation(models:dict, n_points=2048, percentage_kept=0.66):
	label_counter = {model: 1 for model in models.values()}
	with tqdm(
			total=len(models.items()), unit="Files", unit_scale=False, desc="Generating Point clouds", leave=True
	) as pbar:
		for model_path, label in models.items():

			mesh = trimesh.load_mesh(model_path)

			v_np = mesh.vertices.astype(np.float32)
			f_np = mesh.faces.astype(np.int64)
			f_i, bc = pcu.sample_mesh_poisson_disk(v_np, f_np, n_points)
			v_sampled = pcu.interpolate_barycentric_coords(f_np, f_i, bc, v_np)

			noise_mask = torch.rand(v_sampled.shape[0], device="cpu") < percentage_kept

			pcu.save_mesh_v(f"{dataset_dir}/{label}{label_counter[label]}.ply", v_sampled[noise_mask])
			label_counter[label] += 1
			pbar.update(1)

def sample_points_trimesh(models:dict, n_points=2048, percentage_kept=0.66):

	label_counter = {model: 1 for model in models.values()}
	for model_path, label in models.items():
		mesh = trimesh.load_mesh(model_path)
		start = time.time()
		points = mesh.sample(n_points)

		noise_mask = torch.rand(points.shape[0], device="cpu") < percentage_kept
		pcu.save_mesh_v(f"{dataset_dir}/{label}{label_counter[label]}.ply", points[noise_mask])

		label_counter[label] += 1
		end = time.time()
		print(f"Time elapsed: {end - start}")

if __name__ == '__main__':
	#download_from_objaverse()
	#point_cloud_from_camera_view([OBJAVERSE], show_first_cloud=False, percent_saved=0.50)
	number_sample = 2048
	models_3d = get_models_from_datasets([GSO])
	#point_cloud_from_mesh(models_3d, show_first_cloud=False, normalise=False, percent_saved=0.5, n_points=number_sample, points_at_once=50_000)
	pcu_based_generation(models_3d, number_sample)
	#unpack_gso()




