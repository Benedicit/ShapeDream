import os, torch, csv, re
import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
from tqdm import tqdm
import warnings
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing
from collections import defaultdict
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# --- Configuration & Helpers ---
METRICS = ["CD", "UHD", "F-Score", "TMD", "Hausdorff", "NormAgr", "NormCon", "Smooth"]

working_dir = os.path.dirname(os.path.abspath(__file__))

def get_mesh_data(path, device):
    """Loads and normalizes mesh to unit bounding box."""
    if path.endswith(".ply"):
        v, f = load_ply(path)
        from pytorch3d.structures import Meshes
        mesh = Meshes(verts=[v], faces=[f]).to(device)
    else:
        mesh = load_objs_as_meshes([path], device=device, load_textures=False)
    
    v = mesh.verts_packed()
    v = (v - v.mean(0)) / (v.max(0)[0] - v.min(0)[0]).max()
    return mesh.update_padded(v.unsqueeze(0))

def sample_mesh(mesh, n, method="random"):
    if method == "poisson":
        # Fallback to Open3D for Poisson as PyTorch3D lacks native Poisson
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(mesh.verts_packed().cpu().numpy())
        m.triangles = o3d.utility.Vector3iVector(mesh.faces_packed().cpu().numpy())
        pcd = m.sample_points_poisson_disk(n)
        if len(pcd.points) < n: pcd = m.sample_points_uniformly(n) # Fallback
        return torch.from_numpy(np.asarray(pcd.points)).float().to(mesh.device).unsqueeze(0)
    return sample_points_from_meshes(mesh, n, return_normals=True)

# --- Core Evaluator ---
class ShapeEvaluator:
    def __init__(self, device="cuda", method="random"):
        self.device, self.method = device, method

    @torch.no_grad()
    def evaluate(self, p_path, g_path, n=10000, t=0.02):
        m_p, m_g = get_mesh_data(p_path, self.device), get_mesh_data(g_path, self.device)
        
        # Sampling
        if self.method == "poisson":
            p_p, n_p = sample_mesh(m_p, n, "poisson"), torch.zeros((1, n, 3)) # Normals simplified
            p_g, n_g = sample_mesh(m_g, n, "poisson"), torch.zeros((1, n, 3))
            p_p, n_p = sample_mesh(m_p, n)
            p_g, n_g = sample_mesh(m_g, n)
        else:
            p_p, n_p = sample_mesh(m_p, n)  
            p_g, n_g = sample_mesh(m_g, n)
        # Distances
        k_f, k_b = knn_points(p_p, p_g), knn_points(p_g, p_p)
        d_f, d_b = k_f.dists.sqrt(), k_b.dists.sqrt()
        
        # F-Score
        prec, rec = (d_f < t).float().mean(), (d_b < t).float().mean()
        
        return {
            "CD": (d_f.pow(2).mean() + d_b.pow(2).mean()).item(),
            "UHD": max(d_f.max(), d_b.max()).item(),
            "F-Score": (2 * prec * rec / (prec + rec + 1e-8)).item(),
            "TMD": (d_f.mean() + d_b.mean()).item(),
            "Hausdorff": pcu.hausdorff_distance(p_p[0].cpu().numpy(), p_g[0].cpu().numpy()),
            "NormAgr": torch.abs((n_p * torch.gather(n_g, 1, k_f.idx.expand(-1,-1,3))).sum(-1)).mean().item(),
            "NormCon": mesh_normal_consistency(m_p).item(),
            "Smooth": mesh_laplacian_smoothing(m_p).item()
        }

# --- Execution ---
if __name__ == "__main__":
    evaluator = ShapeEvaluator()
    results = defaultdict(list)
    
    # Load paths (Simplified logic)
    gt_map = {row[0]: row[1] for row in csv.reader(open( f"{working_dir}/shapenet_label_to_mesh.csv")) if row}
    #obj_names = [l.strip() for l in open("/home/bweiss/Benedikt/ShapeFormer/demo/dataset/demo.lst") if l.strip() and not l.startswith("#")]
    obj_names = []
    for j in range(4200, 4500):
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

        # Evaluate ShapeFormer (SF) & MVDream (MV)
        #sf_paths = [f"/home/bweiss/Benedikt/ShapeFormer/experiments/demo_shapeformer/meshes_all/{i}_s{s}_mesh.ply" for s in range(3)]
        #sf_res = [evaluator.evaluate(p, gt_p) for p in sf_paths if os.path.exists(p)]
        
        #if sf_res:
        #    results[f"{cls}_ShapeFormer"].append({k: np.mean([r[k] for r in sf_res]) for k in METRICS})
            
        mv_p = f"{working_dir}/../mvdream_2D/debug/{name}/mesh.obj"
        if os.path.exists(mv_p):
            results[f"{cls}_ShapeDream"].append(evaluator.evaluate(mv_p, gt_p))
        interleaved_p = f"{working_dir}/../mvdream_2D/debug2/{name}/mesh.obj"
        
        if os.path.exists(interleaved_p):
            results[f"{cls}_ShapeDreamInterLeaved"].append(evaluator.evaluate(interleaved_p, gt_p))
    # Print Summary
    for k, v in results.items():
        m = {met: np.mean([x[met] for x in v]) for met in METRICS}
        print(f"{k[:25]:<25} | n={len(v)} | " + " | ".join(f"{met}: {m[met]:.8f}" for met in METRICS))

    out_csv = f"{working_dir}/shape_eval_results.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Method", "N"] + METRICS)
        
        for k, v in results.items():
            if not v:
                continue
            m = {met: np.mean([x[met] for x in v]) for met in METRICS}
            writer.writerow(
                [k, len(v)] + [f"{m[met]:.8f}" for met in METRICS]
            )

    print(f"Saved results to {out_csv}")