import glob
import os
import time
import csv
import point_cloud_utils as pcu
from pathlib import Path
import numpy as np
import pytorch3d.structures
import trimesh
import torch
from pytorch3d.io import load_objs_as_meshes
from torch.utils.data import DataLoader, Subset
from pytorch3d.datasets import (ShapeNetCore, collate_batched_meshes)
from pytorch3d.structures import Meshes
from tqdm import tqdm
from mvdream_2D.scripts.trainer import get_mesh_from_pc
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
dataset_dir_masked = working_dir + "/dataset_masked"
dataset_dir_unmasked = working_dir + "/dataset"
RECURSIVE_FILE_PATHS = ("**/*.glb", "**/*.gltf", "**/*.obj", "**/*.ply", "**/*.stl")
generator = np.random.default_rng(42)

def sample_pointcloud_list(pointclouds:list, number_samples=2048, percentage_kept=0.66, masking=False):
	for pc in pointclouds:
		mesh_path = get_mesh_from_pc(pc)
		mesh = load_objs_as_meshes([mesh_path], load_textures=False)
		verts = mesh.verts_packed().cpu().numpy()
		faces = mesh.faces_packed().cpu().numpy()
		v_kept = pcu_based_evenly_spaced_sampling(verts, faces, n_points=number_samples, percentage_kept=percentage_kept, masking=masking)
		pcu.save_mesh_v(f"{dataset_dir_unmasked}/shapenet_chair1500.ply", v_kept)
def shapenet_collate_fn(batch):
	"""
	    Custom collate function to handle ShapeNetCore data:
	    1. Filters out None samples (failed loads).
	    3. Handles missing (None) labels by replacing them with 'unknown'.
	    """
	# 1. Filter out failed loads (None)
	batch = [b for b in batch if b is not None]
	if len(batch) == 0:
		return None

	verts_list = [b["verts"] for b in batch]
	faces_list = [b["faces"] for b in batch]

	# 4. Process metadata keys
	synset_ids = [b["synset_id"] for b in batch]
	model_ids = [b["model_id"] for b in batch]

	# Handle potentially missing labels
	labels = []
	for b in batch:
		lbl = b.get("label")  # safely get label
		labels.append(lbl if lbl is not None else "unknown")

	# 5. Return the clean batch dictionary
	return {
		"verts": verts_list,
		"faces": faces_list,
		"synset_id": synset_ids,
		"model_id": model_ids,
		"label": labels
	}


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
def sample_shapenet(number_samples=2048, masking=True, full_shapenet=True, samples_per_category=150):
	saving_dir = dataset_dir_masked if masking else dataset_dir_unmasked
	shapenet_dir = working_dir + "/.shapenet"
	dataset = ShapeNetCore(shapenet_dir, version=2, load_textures=False)


	if not full_shapenet:
	# 2. Filter indices: Get first 150 of each category
		selected_indices = []
		category_counts = {}

		# dataset.synset_ids maps 1:1 to the loaded models
		for idx, synset_id in enumerate(dataset.synset_ids):
			count = category_counts.get(synset_id, 0)

			if count < samples_per_category:
				selected_indices.append(idx)
				category_counts[synset_id] = count + 1

		# 3. Create the Subset and DataLoader
		dataset = Subset(dataset, selected_indices)

	loader = DataLoader(
		dataset,
		batch_size=1, # We don't actually need the batches as the sampling is done on the cpu
		shuffle=False,
		collate_fn=shapenet_collate_fn,
	)
	label_counter = {}
	csv_path = Path(working_dir + "/shapenet_label_to_mesh.csv")
	with tqdm(
			total=len(loader), unit="Point Clouds", unit_scale=False, desc="Generating Point clouds of ShapeNet objects",
			leave=True
	) as pbar:
		with open(csv_path, "w", newline="") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["label", "filename"])
			for batch in loader:
				if batch is None:
					pbar.update(1)
					continue
				label = batch["label"][0]
				model_id = batch["model_id"][0]
				synset_id = batch["synset_id"][0]
				model_path = os.path.join(shapenet_dir, synset_id, model_id) + "/models/model_normalized.obj"
				if label not in label_counter:
					label_counter[label] = 1
				else:
					label_counter[label] += 1
				v = batch["verts"][0].cpu().numpy()
				f = batch["faces"][0].cpu().numpy()

				writer.writerow([f"shapenet_{label}{label_counter[label]}.ply", model_path])

				v_kept = pcu_based_evenly_spaced_sampling(v, f, n_points=number_samples)
				pcu.save_mesh_v(f"{saving_dir}/shapenet_{label}{label_counter[label]}.ply", v_kept)
				pbar.update(1)

def sample_gso_objaverse(number_samples=2048, masking=True):
	saving_dir = dataset_dir_masked if masking else dataset_dir_unmasked
	models = get_models_from_gso_objaverse([GSO])
	label_counter = {model: 1 for model in models.values()}
	csv_path = Path(working_dir + "/gso_label_to_mesh.csv")
	with tqdm(
			total=len(models.items()), unit="Files", unit_scale=False, desc="Generating Point clouds", leave=True
	) as pbar:
		with open(csv_path, "w", newline="") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["label", "filename"])
			for model_path, label in models.items():
				mesh = trimesh.load_mesh(model_path)

				v_np = mesh.vertices.astype(np.float32)
				f_np = mesh.faces.astype(np.int64)

				v_kept = pcu_based_evenly_spaced_sampling(v_np, f_np, n_points=number_samples, masking=masking)

				pcu.save_mesh_v(f"{saving_dir}/{label}{label_counter[label]}.ply", v_kept)
				writer.writerow([f"{label}{label_counter[label]}.ply", model_path])
				label_counter[label] += 1
				pbar.update(1)

def pcu_based_evenly_spaced_sampling(v, f, n_points=2048, percentage_kept=1.0, masking: bool = True, data_augmentation = True):


			f_i, bc = pcu.sample_mesh_poisson_disk(v, f, n_points, radius=-0.01, random_seed=42) if not masking else pcu.sample_mesh_random(v,f,n_points,random_seed=42)
			v_sampled = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

			if masking:
				if data_augmentation:
					if generator.random() < 0.5:
						split_axis = 0
					else:
						split_axis = 2
					offset = generator.random() * 0.015 - 0.01
				else:
					split_axis = 0
					offset = -0.01

				x_coord_mask = v_sampled[:, split_axis] > offset
				v_filtered = v_sampled[x_coord_mask]
				noise_mask = generator.random(v_filtered.shape[0]) < percentage_kept
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

			pcu.save_mesh_v(f"{dataset_dir_masked}/{label}{label_counter[label]}.ply", points[noise_mask])
			label_counter[label] += 1
			pbar.update(1)

if __name__ == '__main__':

	#sample_gso_objaverse()
	#sample_shapenet(full_shapenet=False)
	sample_pointcloud_list((["shapenet_chair1500.ply"]))

