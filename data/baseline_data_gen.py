import os
import csv
import re
from tqdm import tqdm
from mvdream_2D.scripts.util import get_mesh_from_pc, count_label_entries, load_pcd_to_tensor
from mvdream_2D.scripts.view_renderer import normalize_vertices
from pytorch3d.io import load_objs_as_meshes, load_ply, save_ply
from pytorch3d.ops import knn_points, sample_points_from_meshes
from lightning import seed_everything
import random
import torch
import h5py
import mvdream_2D.scripts.gen_partial_pc

from pathlib import Path

from scripts.gen_partial_pc import generate_partial_shapes

input_pc_pcn_path = f"/home/stud/weisb/ShapeDream/data/.pcn/ShapeNetCompletion/val/partial/03001627"
basepath_pcn = f"/home/stud/weisb/ShapeDream/data/.pcn/ShapeNetCompletion/val/complete/03001627"

working_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    seed_everything(42)
    """
    class_names = [# "airplane",
                   # "bag",
                   # "basket",
                   # "bathtub",
                   # "bed",
                   # "bench",
                   # "birdhouse",
                   # "bookshelf",
                   # "bottle",
                   # "bowl",
                   # "bus",
                   # "cabinet",
                   # "camera",
                   # "can",
                   # "cap",
                   # "car",
                   # "cellphone",
                   "chair",
                   # "clock",
                   # "dishwasher",
                   # "display",
                   # "earphone",
                   # "faucet",
                   # "file cabinet",
                   # "flowerpot",
                   # "guitar",
                   # "helmet",
                   # "jar",
                   # "keyboard",
                   # "knife",
                   # "lamp",
                   # "laptop",
                   # "loudspeaker",
                   # "mailbox",
                   # "microphone",
                   # "microwave",
                   # "motorbike",
                   # "mug",
                   # "piano",
                   # "pillow",
                   # "pistol",
                   # "printer",
                   # "remote",
                   # "rifle",
                   # "rocket",
                   # "skateboard",
                   # "sofa",
                   # "stove",
                   # "table",
                   # "telephone",
                   # "tower",
                   # "train",
                   # "trash bin",
                   # "washer",
                   # "watercraft",
                   ]
    gt_map = {row[0]: row[1] for row in csv.reader(open( f"{working_dir}/shapenet_label_to_mesh.csv")) if row}
    obj_names = []
    grid_size = 64

    # Pre-compute the flat uniform voxel grid grid once [1, 262144, 3]
    lin_space = torch.linspace(-1.0, 1.0, grid_size, device="cuda")
    grid_x, grid_y, grid_z = torch.meshgrid(lin_space, lin_space, lin_space, indexing="ij")
    grid_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, -1, 3)

    for cl in class_names:
        num_obj = count_label_entries(cl)
        start = min(int(0.7 * num_obj) + 1, 4250)
        for j in range(start, min(num_obj, start + 250)):
            obj_names.append(f"{cl}{j}")

        for i, name in enumerate(tqdm(obj_names)):
            cls = re.match(r"^[A-Z a-z]+", name).group(0).lower()
            obj_path = gt_map.get(f"shapenet_{name}")
            if not obj_path:
                continue
            os.makedirs(f"{working_dir}/ground_truth/{cls}", exist_ok=True)
            os.makedirs(f"{working_dir}/input_pc/{cls}", exist_ok=True)
            obj = load_objs_as_meshes([obj_path], load_textures=False, device="cuda")
            gt_pc, gt_normals = sample_points_from_meshes(obj, num_samples=16384, return_normals=True)

            centroid = gt_pc.mean(dim=1, keepdim=True)
            gt_pc_centered = gt_pc - centroid
            max_dist = torch.sqrt((gt_pc_centered ** 2).sum(dim=2)).max(dim=1, keepdim=True)[0]

            #gt_pc = normalize_vertices(gt_pc)
            gt_pc = gt_pc_centered / max_dist.unsqueeze(2)
            gt_pc = gt_pc.squeeze(0)

            save_ply(f"{working_dir}/ground_truth/{cls}/{name}.ply", gt_pc)

            input_pc = sample_points_from_meshes(obj, num_samples=8192)

            input_pc_centered = input_pc - centroid
            input_pc = input_pc_centered / max_dist.unsqueeze(2)

            B_pts, N_pool, _ = input_pc.shape
            split_axis = 0 if random.random() < 0.5 else 2
            offset = random.random() * 0.015
            percentage_kept = 0.75

            # Compute masks for the whole batch at once
            axis_mask = input_pc[..., split_axis] > offset
            dropout_mask = torch.rand((B_pts, N_pool), device="cuda") < (4096 / N_pool * percentage_kept)
            combined_mask = axis_mask & dropout_mask # [B, N]

            input_pc = input_pc[combined_mask]
            input_pc = input_pc.squeeze(0)

            save_ply(f"{working_dir}/input_pc/{cls}/{name}.ply", input_pc)
    train_samples = list(Path(input_pc_pcn_path).glob("**/*.pcd"))
    train_samples = sorted(train_samples)
    for sample in tqdm(train_samples):
        flat_points = load_pcd_to_tensor(sample)
        idx = torch.randperm(flat_points.size(0))[:500]
        sparse_points = flat_points[idx]
        #sparse_points = flat_points
        save_ply(f"{working_dir}/input_pc/chair/{sample.parent.name}.ply", sparse_points)
    """
    base_path = f"/home/stud/weisb/ShapeDream/data/.pcn/ShapeNetCompletion/val/complete/03001627"
    gt_paths = list(Path(base_path).glob("**/*.pcd"))
    gt_paths = sorted(gt_paths)
    progressbar = tqdm(gt_paths)
    point_clouds = dict()
    for p in gt_paths:
        name = p.stem
        mesh_path = (
                Path(working_dir) / "../data" / ".shapenet"
                / "03001627" / name
                / "models" / "model_normalized.obj"
        )
        mesh = load_objs_as_meshes([mesh_path], load_textures=False, device="cuda")
        gt_pc = sample_points_from_meshes(mesh, num_samples=16384)
        point_clouds[name] = gt_pc.squeeze(0)
        progressbar.update(1)
    generate_partial_shapes(point_clouds,f"{working_dir}/input_pc/chair", use_fps=True)


