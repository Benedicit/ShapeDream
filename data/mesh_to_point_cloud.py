import glob
import os
import time
import csv
import point_cloud_utils as pcu
import torch
import numpy as np
import trimesh
import torch
from torch.utils.data import DataLoader, Subset
from pytorch3d.datasets import ShapeNetCore

from tqdm import tqdm

OBJAVERSE = "objaverse"
REDWOOD = "redwood"
GSO = "gso"
SHAPENET = "shapenet"

working_dir = os.path.dirname(os.path.realpath(__file__))
original_datasets_dirs = {
	OBJAVERSE: "/.objaverse",
	REDWOOD: "/.redwood",
	GSO: "/.gso",
	SHAPENET: "/.shapenet"
}
dataset_dir = working_dir + "/dataset"
RECURSIVE_FILE_PATHS = ("**/*.glb", "**/*.gltf", "**/*.obj", "**/*.ply", "**/*.stl")

def get_models_from_gso_objaverse(datasets: list):
	models = []
	labels = dict()
	for d in datasets:
		if d not in original_datasets_dirs:
			continue
		if d == GSO:
			with open(f"{working_dir}/{GSO}_labels.csv", encoding="utf-8") as f:
				temp = {row["filename"]: row["label"] for row in csv.DictReader(f)}
				labels.update(temp)
		if d == SHAPENET:
			print("ShapeNet is handled seperatetly using the Pytorch3D Dataloader")
			pass
		current_dir = working_dir + original_datasets_dirs[d]
		file_patterns = [rf"{current_dir}/{suffix}" for suffix in
						 RECURSIVE_FILE_PATHS]
		for pattern in file_patterns:
			models += glob.glob(pattern, recursive=True)
	labels = {m: labels[os.path.basename(m)] for m in models if os.path.basename(m) in labels}

	return labels
def sample_shapenet(number_samples=2048):
	shapenet_dir = working_dir + "/.shapenet"
	dataset = ShapeNetCore(shapenet_dir, version=2, load_textures=True)

	# 2. Filter indices: Get first 100 of each category
	selected_indices = []
	category_counts = {}
	limit_per_category = 2

	# dataset.synset_ids maps 1:1 to the loaded models
	for idx, synset_id in enumerate(dataset.synset_ids):
		count = category_counts.get(synset_id, 0)

		if count < limit_per_category:
			selected_indices.append(idx)
			category_counts[synset_id] = count + 1

	# 3. Create the Subset and DataLoader
	subset_dataset = Subset(dataset, selected_indices)

	# Use ShapeNetCore.collate_fn to handle batches of Meshes with different topologies
	loader = DataLoader(
		subset_dataset,
		batch_size=1, # We don't need the batches
		shuffle=False,
	)

	label_counter = {}
	with tqdm(
			total=len(loader), unit="Point Clouds", unit_scale=False, desc="Generating Point clouds of ShapeNet objects",
			leave=True
	) as pbar:
		for batch in loader:
			label = batch["label"][0]
			if label not in label_counter:
				label_counter[label] = 1
			else:
				label_counter[label] += 1
			v = batch["verts"][0].cpu().numpy()
			f = batch["faces"][0].cpu().numpy()
			v_kept = pcu_based_evenly_spaced_sampling(v, f, n_points=number_samples)
			pcu.save_mesh_v(f"{dataset_dir}/shapenet_{label}{label_counter[label]}.ply", v_kept)
			pbar.update(1)


def sample_gso_objaverse(number_samples=2048):
	models = get_models_from_gso_objaverse([GSO])
	label_counter = {model: 1 for model in models.values()}
	with tqdm(
			total=len(models.items()), unit="Files", unit_scale=False, desc="Generating Point clouds", leave=True
	) as pbar:
		for model_path, label in models.items():
			mesh = trimesh.load_mesh(model_path)

			v_np = mesh.vertices.astype(np.float32)
			f_np = mesh.faces.astype(np.int64)
			v_kept = pcu_based_evenly_spaced_sampling(v_np, f_np, n_points=number_samples)

			pcu.save_mesh_v(f"{dataset_dir}/{label}{label_counter[label]}.ply", v_kept)
			label_counter[label] += 1
			pbar.update(1)

def pcu_based_evenly_spaced_sampling(v, f, n_points=2048, percentage_kept=0.66, masking: bool = True):

			f_i, bc = pcu.sample_mesh_poisson_disk(v, f, n_points)
			v_sampled = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

			if masking:
				x_coord_mask = v_sampled[:, 0] > -0.01
				v_filtered = v_sampled[x_coord_mask]
				noise_mask = np.random.random(v_filtered.shape[0]) < percentage_kept
				v_kept = v_filtered[noise_mask]
			else:
				v_kept = v_sampled
			return v_kept

def sample_points_trimesh(models:dict, n_points=2048, percentage_kept=0.66):

	label_counter = {model: 1 for model in models.values()}
	with tqdm(
			total=len(models.items()), unit="Files", unit_scale=False, desc="Generating Point clouds", leave=True
	) as pbar:
		for model_path, label in models.items():

			mesh = trimesh.load_mesh(model_path)
			points = mesh.sample(n_points)

			noise_mask = torch.rand(points.shape[0], device="cpu") < percentage_kept

			pcu.save_mesh_v(f"{dataset_dir}/{label}{label_counter[label]}.ply", points[noise_mask])
			label_counter[label] += 1
			pbar.update(1)

if __name__ == '__main__':

	#sample_gso_objaverse()
	sample_shapenet()




