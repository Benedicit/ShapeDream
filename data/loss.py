import os, torch, csv, re

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
from tqdm import tqdm
import warnings
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing, chamfer_distance
from collections import defaultdict
from mvdream_2D.scripts.view_renderer import normalize_vertices

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# --- Configuration & Helpers ---
METRICS = ["CD", "UHD", "F-Score", "TMD", "Hausdorff", "NormAgr", "NormCon", "Smooth"]
METRICS_PYTORCH = ["CD_L1", "CD_L2", "UHD", "F-Score"]

working_dir = os.path.dirname(os.path.abspath(__file__))

# --- Core Evaluator ---
class ShapeEvaluator:
    def __init__(self, device="cuda", method="random"):
        self.device, self.method = device, method

    @torch.no_grad()
    def evaluate(self, p_path, g_path, num_points=16384, threshold=0.0001):
        # load_objs_as_meshes expects a list of files/paths
        g_mesh = load_objs_as_meshes(g_path, device=self.device, load_textures=False)
        p_mesh = load_objs_as_meshes(p_path, device=self.device, load_textures=False)

        # Sample surface points
        g_points = sample_points_from_meshes(g_mesh, num_points)
        p_points = sample_points_from_meshes(p_mesh, num_points)

        g_points = normalize_vertices(g_points)
        p_points = normalize_vertices(p_points)

        # Chamfer distances
        cd_l1, _ = chamfer_distance(p_points, g_points, norm=1)
        cd_l2, _ = chamfer_distance(p_points, g_points, norm=2)
        uhd, _ = chamfer_distance(p_points, g_points, norm=2, single_directional=True, point_reduction="max")

        # F-score
        knn_p2g = knn_points(p_points, g_points, K=1)
        knn_g2p = knn_points(g_points, p_points, K=1)

        d_p2g = knn_p2g.dists[..., 0]
        d_g2p = knn_g2p.dists[..., 0]

        # NOTE: treshhold is 0.0001 by default which is the F1-score@1% as the knn_points are the squared distances
        prec = (d_p2g < threshold).float().mean(dim=1)
        rec = (d_g2p < threshold).float().mean(dim=1)
        f_score = 2 * prec * rec / (prec + rec + 1e-8)

        f_score_mean = f_score.mean()

        return {
            "CD_L1": cd_l1.detach().cpu().item(),
            "CD_L2": cd_l2.detach().cpu().item(),
            "UHD": uhd.detach().cpu().item(),
            "F-Score": f_score_mean.detach().cpu().item(),
        }

if __name__ == "__main__":
    evaluator = ShapeEvaluator()
    results = defaultdict(list)
    
    # Load paths
    gt_map = {row[0]: row[1] for row in csv.reader(open( f"{working_dir}/shapenet_label_to_mesh.csv")) if row}
    obj_names = []
    for j in range(4250, 4500):
        obj_names.append(f"chair{j}")

    for i, name in enumerate(tqdm(obj_names)):
        cls = re.match(r"^[A-Z a-z]+", name).group(0).lower()
        gt_p_raw = gt_map.get(f"shapenet_{name}.ply")
        
        if not gt_p_raw:
            continue

        # If the path in the CSV is the directory, append the standard ShapeNet model path
        if os.path.isdir(gt_p_raw):
            gt_p = os.path.join(gt_p_raw, "models", "model_normalized.obj")
        else:
            gt_p = gt_p_raw

        if not os.path.exists(gt_p):
            print(f"Warning: File not found {gt_p}")
            continue

        mv_p = f"{working_dir}/../mvdream_2D/scripts/debug/{name}/mesh.obj"
        if os.path.exists(mv_p):
            results[f"{cls}_ShapeDream"].append(evaluator.evaluate([mv_p], [gt_p]))
        interleaved_p = f"{working_dir}/../mvdream_2D/scripts/debug2/{name}/mesh.obj"
        
        if os.path.exists(interleaved_p):
            results[f"{cls}_ShapeDream_no_text"].append(evaluator.evaluate([interleaved_p], [gt_p]))

    # Print Summary
    for k, v in results.items():
        m = {met: np.mean([x[met] for x in v]) for met in METRICS_PYTORCH}
        print(f"{k[:25]:<25} | n={len(v)} | " + " | ".join(f"{met}: {m[met]:.8f}" for met in METRICS_PYTORCH))
        print(k)

    out_csv = f"{working_dir}/shape_eval_results.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Method", "N"] + METRICS_PYTORCH)
        #writer.writerow(["Method", "N"] + METRICS)

        for k, v in results.items():
            if not v:
                continue
            m = {met: np.mean([x[met] for x in v]) for met in METRICS_PYTORCH}
            #m = {met: np.mean([x[met] for x in v]) for met in METRICS}
            writer.writerow(
                [k, len(v)] + [f"{m[met]:.8f}" for met in METRICS_PYTORCH]
                #[k, len(v)] + [f"{m[met]:.8f}" for met in METRICS]
            )

    print(f"Saved results to {out_csv}")