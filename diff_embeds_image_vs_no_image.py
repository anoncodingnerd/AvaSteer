# %%
import os
import os.path as op
from typing import List
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from utils import construct_diff_df
import json
import re
import pandas as pd
from matplotlib import font_manager
import inspect
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

# %%
EMBEDS_A_DIR = "Data/Representation/MathV_split/Qwen3-VL-8B-Thinking-no-image"
EMBEDS_B_DIR = "Data/Representation/MathV_split/Qwen3-VL-8B-Thinking"
#EMBEDS_A_DIR = "Data/Representation/MathV_split/GLM-4.1V-9B-Thinking-no-image"
#EMBEDS_B_DIR = "Data/Representation/MathV_split/GLM-4.1V-9B-Thinking"
NUM_LAYERS = 40
DIRECTION = "a_minus_b"  # "a_minus_b" or "b_minus_a"

# %%
OUT_DIR = "Data/Representation/Diff_split_image_vs_no_image_8b"
# OUT_DIR = "Data/Representation/Diff_MMMU_GLM_image_vs_no_image"


def load_layer_paths(root: str, num_layers: int) -> List[str]:
    paths: List[str] = []
    for idx in range(num_layers):
        path = op.join(root, f"embeds_{idx}.npy")
        if not op.exists(path):
            raise FileNotFoundError(f"Missing layer file: {path}")
        paths.append(path)
    return paths



# %%

def main() -> None:
    if DIRECTION not in {"a_minus_b", "b_minus_a"}:
        raise ValueError(f"Unsupported direction: {DIRECTION}")

    os.makedirs(OUT_DIR, exist_ok=True)

    paths_a = load_layer_paths(EMBEDS_A_DIR, NUM_LAYERS)
    paths_b = load_layer_paths(EMBEDS_B_DIR, NUM_LAYERS)

    for idx, (path_a, path_b) in enumerate(zip(paths_a, paths_b)):
        arr_a = np.load(path_a)
        arr_b = np.load(path_b)
        if arr_a.shape != arr_b.shape:
            raise ValueError(
                f"Shape mismatch at layer {idx}: {arr_a.shape} vs {arr_b.shape}"
            )
        if DIRECTION == "a_minus_b":
            diff = arr_a - arr_b
        else:
            diff = arr_b - arr_a
        out_path = op.join(OUT_DIR, f"embeds_{idx}.npy")
        np.save(out_path, diff)

    print(f"[info] saved diffs to: {OUT_DIR}")


if __name__ == "__main__":
    main()


# %%
for i in range(NUM_LAYERS):
    path = os.path.join(OUT_DIR, f"embeds_{i}.npy")
    diff = np.load(path)
    per_sample = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    mean_norm = per_sample.mean()
    std_norm = per_sample.std()
    frob_norm = np.linalg.norm(diff)
    print(f"layer {i:02d} | mean={mean_norm:.4f} std={std_norm:.4f} frob={frob_norm:.4f}")

# %%
def cosine_to_mean(mat: np.ndarray):
    # mat: [N, H] or [N, ...]
    mat2d = mat.reshape(mat.shape[0], -1)
    mean_vec = mat2d.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm == 0:
        return None, None
    mean_unit = mean_vec / mean_norm
    vec_norms = np.linalg.norm(mat2d, axis=1)
    valid = vec_norms > 0
    cos = (mat2d[valid] @ mean_unit) / vec_norms[valid]
    return cos, mean_norm

mean_vec_norm = [0, ]
cos_mean = [0, ]
cos_std = [0, ]
layers = [0, ]

for i in range(NUM_LAYERS):
    path = os.path.join(OUT_DIR, f"embeds_{i}.npy")
    diff = np.load(path)
    cos, mean_norm = cosine_to_mean(diff)
    if cos is None:
        print(f"layer {i:02d} | mean_vec_norm=0")
        continue
    
    mean_vec_norm.append(mean_norm)
    cos_mean.append(cos.mean())
    cos_std.append(cos.std())
    layers.append(i)

    print(
        f"layer {i:02d} | mean_vec_norm={mean_norm:.4f} "
        f"| cos_mean={cos.mean():.4f} std={cos.std():.4f} "
        f"| same_dir_ratio={(cos>0).mean():.3f} "
        f"| socre={cos.mean()/cos.std():.4f}"
    )

# %%
print(mean_vec_norm)
print(cos_mean)
print(cos_std)
print(layers)

# %%

font_path = "Assets/Times New Roman Bold.ttf"
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)

# y1: Cosine Similarity Mean
ax1.plot(layers, cos_mean, color="#4C78A8", linestyle='--', label='Cosine Similarity Mean', linewidth=2.5)
ax1.set_ylabel('Cosine Similarity Mean', fontsize=16)
ax1.tick_params(axis='y')

# y2: Cosine Similarity Std
ax2 = ax1.twinx()
ax2.plot(layers, cos_std, color="#F58518", linestyle=':', label='Cosine Similarity Standard Deviation', linewidth=2.5)
ax2.set_ylabel('Cosine Similarity Standard Deviation', fontsize=16)
ax2.tick_params(axis='y')

ax1.set_xlabel('Layers', fontsize=16)
ax1.set_title('GLM-4.1V-9B-Thinking Layer Analysis', fontsize=16)

lines = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# %%
mat2d = diff.reshape(diff.shape[0], -1)
mean_vec = mat2d.mean(axis=0)
mean_unit = mean_vec / np.linalg.norm(mean_vec)
cos = (mat2d @ mean_unit) / np.linalg.norm(mat2d, axis=1)

print("min_cos", cos.min(), "p01", np.quantile(cos, 0.01))
print("mean_norm/avg_norm", np.linalg.norm(mean_vec) / np.mean(np.linalg.norm(mat2d, axis=1)))


# %%
neg = (cos <= 0).sum()
total = cos.size
ratio = 1 - neg / total
print("neg", int(neg), "total", int(total), "ratio", ratio)
print("ratio_6dp", f"{ratio:.6f}", "min", cos.min(), "p01", np.quantile(cos, 0.01))


# %%
### Save Steering Vector

out_path = "Data/Representation/SPLIT_steering_vectors_image_vs_no_image_glm.npy"
direction = "a_minus_b"
# direction = "b_minus_a"
normalize = False

layer_vecs = []
for i in range(1, NUM_LAYERS + 1):
    a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{i}.npy"))
    b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{i}.npy"))
    diff = a - b if direction == "a_minus_b" else b - a
    vec = diff.mean(axis=0)  # [H]
    if normalize:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    layer_vecs.append(vec)

steering = np.stack(layer_vecs, axis=0)  # [L, H]
np.save(out_path, steering)
print("saved:", out_path, steering.shape)

# %%

# Use one representation set from MathV_split (not A-B diff).
REPRESENTATION_DIR = EMBEDS_B_DIR  # e.g. with-image; switch to EMBEDS_A_DIR for no-image
MATHV_JSONL = "Data/Questions/MathV.jsonl"
MATHV_MINI_JSONL = "Data/Questions/MathVMini.jsonl"
DROP_LAST_VECTOR = True  # keep consistent with AnalyzeOneModel.py (save mean_steering_vectors[:-1])

rep_name = os.path.basename(REPRESENTATION_DIR.rstrip("/"))
OUT_PATH = f"Data/Representation/Steering_vectors/MathV_split_{rep_name}_mean_steering_vectors_by_level.npy"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_mathv_split_info(mathv_jsonl, mini_jsonl):
    full_data = read_jsonl(mathv_jsonl)
    mini_data = read_jsonl(mini_jsonl)
    mini_ids = {str(item.get("id")) for item in mini_data}

    split_data = [item for item in full_data if str(item.get("id")) not in mini_ids]
    df = pd.DataFrame(split_data)

    # Keep fields compatible with construct_diff_df.
    df["problem"] = df["id"].astype(str)

    def to_level_label(x):
        if isinstance(x, str):
            return x if x.startswith("Level") else f"Level {x}"
        if pd.isna(x):
            return "Level ?"
        return f"Level {int(x)}"

    df["level"] = df["level"].apply(to_level_label)
    df["avg_tokens_reasoning"] = np.nan
    return df[["problem", "level", "avg_tokens_reasoning"]].copy()


repr_files = sorted(
    [f for f in os.listdir(REPRESENTATION_DIR) if re.match(r"embeds_\d+\.npy$", f)],
    key=lambda x: int(re.search(r"(\d+)", x).group(1)),
)
if not repr_files:
    raise FileNotFoundError(f"No embeds_*.npy found in {REPRESENTATION_DIR}")

base_info = build_mathv_split_info(MATHV_JSONL, MATHV_MINI_JSONL)
n_samples = np.load(os.path.join(REPRESENTATION_DIR, repr_files[0]), mmap_mode="r").shape[0]
if len(base_info) != n_samples:
    raise ValueError(
        f"MathV_split metadata count mismatch: metadata={len(base_info)} vs embeddings={n_samples}."
    )

diff_dfs_by_layer = []
mean_steering_vectors = []

for idx, file in enumerate(repr_files):
    X = np.load(os.path.join(REPRESENTATION_DIR, file))
    df_layer = base_info.copy()
    df_layer["embedding"] = list(X)

    diff_df, level_names = construct_diff_df(df_layer, X)
    diff_dfs_by_layer.append(diff_df)

    # Mean over all level-difference vectors in the current layer.
    mean_vec = np.mean(np.vstack(diff_df["diff_embedding"].values), axis=0)
    mean_steering_vectors.append(mean_vec)

    if idx in {0, len(repr_files) - 1}:
        print(f"layer {idx:02d}: levels={level_names}, mean_norm={np.linalg.norm(mean_vec):.4f}")

mean_steering_vectors = np.asarray(mean_steering_vectors)
to_save = mean_steering_vectors[:-1] if DROP_LAST_VECTOR else mean_steering_vectors

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.save(OUT_PATH, to_save)
print(f"saved: {OUT_PATH}")
print(f"all layers shape: {mean_steering_vectors.shape}, saved shape: {to_save.shape}")


# %%

K_MAX = 40

def load_layer(path):
    return np.load(path)  # shape [N, H]

def pca_cumvar(X, k_max):
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt, singular values S
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / (Xc.shape[0] - 1)
    total = eigvals.sum()
    cumvar = np.cumsum(eigvals) / (total + 1e-12)
    return cumvar[:k_max]

plt.figure(figsize=(6, 4))
for layer in range(1, NUM_LAYERS, 5):
    a = load_layer(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
    b = load_layer(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))
    X = np.concatenate([a, b], axis=0)  # [N_total, H]
    cumvar = pca_cumvar(X, K_MAX)
    plt.plot(range(1, len(cumvar)+1), cumvar, label=f"L{layer}")

plt.xlabel("Number of PCs (k)")
plt.ylabel("Cumulative Variance Ratio")
plt.title("PCA Cumulative Variance by Layer")
plt.legend(ncol=4, fontsize=6)
plt.tight_layout()
plt.show()

# %%
STEER_PATH = "Data/Representation/steering_vectors_image_vs_no_image.npy"
NUM_LAYERS = 36
K = 10  # top-k PCs

def pca_project_ratio(X, r, k):
    # X: [N,H], r: [H]
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Ueff = Vt[:k].T  # [H,k]
    r_proj = Ueff @ (Ueff.T @ r)
    return np.linalg.norm(r_proj) / (np.linalg.norm(r) + 1e-12)

steer = np.load(STEER_PATH)  # [L,H]
ratios = []

for layer in range(NUM_LAYERS):
    a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
    b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))
    X = np.concatenate([a, b], axis=0)
    r = steer[layer]
    ratio = pca_project_ratio(X, r, K)
    ratios.append(ratio)
    print(f"layer {layer:02d} | ratio={ratio:.4f}")

plt.figure(figsize=(5,3))
plt.plot(range(NUM_LAYERS), ratios, marker="o")
plt.xlabel("Layer")
plt.ylabel("||P_M r|| / ||r||")
plt.title(f"Projection Energy Ratio (k={K})")
plt.tight_layout()
plt.show()

# %%

# np.load("Data/Representation/MathV_split/Qwen3-VL-2B-Thinking/calibration_vectors.npy")
np.load("Data/Representation/MathV_split/Qwen3-VL-2B-Thinking/calibration_vectors.npy").min()

# %%
scores = []
for layer in range(NUM_LAYERS):
    a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
    b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    r = mu_a - mu_b
    r_norm = np.linalg.norm(r) + 1e-12
    r_unit = r / r_norm

    proj_a = a @ r_unit
    proj_b = b @ r_unit
    # 分离度 / 噪声
    score = (proj_a.mean() - proj_b.mean()) / (proj_a.std() + proj_b.std() + 1e-12)

    scores.append((layer, score, r_norm))

# 只看后 1/3 层
candidates = [x for x in scores if x[0] >= NUM_LAYERS * 2 // 3]
candidates.sort(key=lambda x: x[1], reverse=True)

print("Top candidates (layer, score, r_norm):")
for item in candidates[:5]:
    print(item)

# %%
def compute_pca_basis(X, k=None, var_thresh=None):
    # X: [N, H]
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig = (S ** 2) / (Xc.shape[0] - 1)
    if var_thresh is not None:
        cum = np.cumsum(eig) / (eig.sum() + 1e-12)
        k = int(np.searchsorted(cum, var_thresh) + 1)
    if k is None:
        raise ValueError("Provide k or var_thresh.")
    Ueff = Vt[:k].T  # [H, k]
    return Ueff

def purify_steering(a, b, k=10, var_thresh=None, eps=1e-12):
    # a, b: [N, H]
    r = a.mean(axis=0) - b.mean(axis=0)  # r(l*)
    Ueff = compute_pca_basis(np.concatenate([a, b], axis=0), k=k, var_thresh=var_thresh)
    r_proj = Ueff @ (Ueff.T @ r)
    norm = np.linalg.norm(r_proj)
    if norm > eps:
        r_proj = r_proj / norm
    return r_proj

layer = 24
a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))

r_purified = purify_steering(a, b, k=10)  # or var_thresh=0.8
np.save("Data/Representation/steering_vector_layer24_purified.npy", r_purified)
print("saved", r_purified.shape)

# %%


root = "Data/Representation/MathV_split/GLM-4.1V-9B-Thinking"

def layer_idx(path):
    m = re.search(r"embeds_(\d+)\.npy$", path)
    return int(m.group(1)) if m else -1

paths = [
    os.path.join(root, f) for f in os.listdir(root)
    if f.startswith("embeds_") and f.endswith(".npy")
]
paths = sorted(paths, key=layer_idx)

means = []
for p in paths:
    arr = np.load(p, mmap_mode="r")  # [N, H]
    means.append(arr.mean(axis=0).astype(np.float32))

calib = np.stack(means, axis=0)  # [L, H]
out_path = os.path.join(root, "calibration_vectors.npy")
np.save(out_path, calib)
print("saved:", out_path, calib.shape)


# %%

font_path = "Assets/Times New Roman Bold.ttf"
font_manager.fontManager.addfont(font_path)
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

def plot_tsne_by_group(df_tsne, model_name=None, figsize=(12, 8), 
                          point_size=50, mean_point_size=200,
                          alpha=0.7, mean_alpha=1.0,
                          save_path=None):
    """
    Visualize the data points and the corresponding mean points using t-SNE.

    Parameters:
        df_tsne: The DataFrame containing the reduced-dimensional data, needs to include the columns:
                 ['tsne_x', 'tsne_y', 'group', 'is_mean']
        model_name: The name of the model, used for the chart title
        figsize: The size of the chart, default is (12, 8)
        point_size: The size of the ordinary data points, default is 50
        mean_point_size: The size of the mean points, default is 200
        alpha: The transparency of the ordinary data points, default is 0.7
        mean_alpha: The transparency of the mean points, default is 1.0
        save_path: The path to save the chart, default is None
    """
    # Set color mapping for different groups.
    unique_groups = sorted(df_tsne[~df_tsne['is_mean']]['group'].unique())
    if len(unique_groups) == 2:
        base_colors = ["#4C78A8", "#F58518"]
    else:
        base_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
    color_map = dict(zip(unique_groups, base_colors))

    # Create a chart.
    plt.figure(figsize=figsize, dpi=300)

    # First, plot the ordinary data points.
    for group in unique_groups:
        mask = (df_tsne['group'] == group) & (~df_tsne['is_mean'])
        plt.scatter(df_tsne.loc[mask, 'tsne_x'],
                   df_tsne.loc[mask, 'tsne_y'],
                   c=[color_map[group]],
                   label=group,
                   alpha=alpha,
                   s=point_size,
                   marker='o',
                   edgecolors='none')

    # Then plot the mean point (double-draw for visibility).
    for group in unique_groups:
        mask = df_tsne['group'] == f'Mean {group}'
        if mask.any():
            plt.scatter(df_tsne.loc[mask, 'tsne_x'],
                       df_tsne.loc[mask, 'tsne_y'],
                       c='white',
                       marker='*',
                       label='_nolegend_',
                       alpha=1.0,
                       s=mean_point_size * 1.8,
                       edgecolors='black',
                       linewidths=2.5,
                       zorder=5)
            plt.scatter(df_tsne.loc[mask, 'tsne_x'],
                       df_tsne.loc[mask, 'tsne_y'],
                       c=[color_map[group]],
                       marker='*',
                       label=f'Mean {group}',
                       alpha=mean_alpha,
                       s=mean_point_size,
                       edgecolors='black',
                       linewidths=1.2,
                       zorder=6)

    # Set the title and tags.
    title = 'Layer 28 (Qwen3-VL-Thinking-8B)'
    #title = 'Layer 30 (GLM-4.1V-9B-Thinking)'
    if model_name:
        title += f'\n{model_name}'
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)

    # Legend inside the plot.
    plt.legend(title='Group',
              loc='upper right',
              frameon=True,
              ncol=2,
              fontsize=11)

    plt.tight_layout()

    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# %%

layer_id = 28
a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{layer_id}.npy"))
b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{layer_id}.npy"))
a = a.reshape(a.shape[0], -1)
b = b.reshape(b.shape[0], -1)

mean_a = a.mean(axis=0, keepdims=True)
mean_b = b.mean(axis=0, keepdims=True)

X = np.vstack([a, b, mean_a, mean_b])
labels = (["w/o Image"] * len(a) + ["w/ Image"] * len(b)
          + ["Mean w/o Image", "Mean w/ Image"])
is_mean = [False] * (len(a) + len(b)) + [True, True]

# PCA pre-reduction helps t-SNE behave more smoothly
if X.shape[1] > 50 and X.shape[0] > 2:
    n_pca = min(50, X.shape[0] - 1, X.shape[1])
    X = PCA(n_components=n_pca, random_state=42).fit_transform(X)

n_samples = X.shape[0]
tsne_kwargs = dict(n_components=2, perplexity=15, learning_rate=300,
                   early_exaggeration=6, init="pca", random_state=42, metric="cosine")
if "n_iter" in inspect.signature(TSNE).parameters:
    tsne_kwargs["n_iter"] = 3000
coords = TSNE(**tsne_kwargs).fit_transform(X)

df_tsne = pd.DataFrame({
    "tsne_x": coords[:, 0],
    "tsne_y": coords[:, 1],
    "group": labels,
    "is_mean": is_mean,
})

plot_tsne_by_group(
    df_tsne,
    model_name=None,
    figsize=(4, 4),
    point_size=30,
    mean_point_size=220,
    alpha=0.6,
    mean_alpha=1.0,
    save_path=None,
)


# %%

text = torch.from_numpy(np.load("Assets/MathV/Qwen3-VL-8B-Thinking/mean_steering_vectors_all.npy"))
text_split = torch.from_numpy(np.load("Data/Representation/Steering_vectors/MathV_split_Qwen3-VL-8B-Thinking_mean_steering_vectors_by_level.npy"))
visual = torch.from_numpy(np.load("Data/Representation/Steering_vectors/SPLIT_steering_vectors_image_vs_no_image_8b.npy"))

cos = torch.nn.functional.cosine_similarity(text_split, visual, dim=1)
print(cos)

# %%
text_4b = torch.from_numpy(np.load("Assets/MathV/Qwen3-VL-4B-Thinking/mean_steering_vectors_all.npy"))
text_split_4b = torch.from_numpy(np.load("Data/Representation/Steering_vectors/MathV_split_Qwen3-VL-4B-Thinking_mean_steering_vectors_by_level.npy"))
visual_4b = torch.from_numpy(np.load("Data/Representation/Steering_vectors/SPLIT_steering_vectors_image_vs_no_image_4b.npy"))

cos_4b = torch.nn.functional.cosine_similarity(text_split_4b, visual_4b, dim=1)
print(cos_4b)

# %%
#text_2b = torch.from_numpy(np.load("Assets/MathV/Qwen3-VL-2B-Thinking/mean_steering_vectors_all.npy"))
text_split_2b = torch.from_numpy(np.load("Data/Representation/Steering_vectors/MathV_split_Qwen3-VL-2B-Thinking_mean_steering_vectors_by_level.npy"))
visual_2b = torch.from_numpy(np.load("Data/Representation/Steering_vectors/SPLIT_steering_vectors_image_vs_no_image_2b.npy"))

cos_2b = torch.nn.functional.cosine_similarity(text_split_2b, visual_2b, dim=1)
print(cos_2b)

# %%
text_split_glm = torch.from_numpy(np.load("Data/Representation/Steering_vectors/MathV_split_GLM-4.1V-9B-Thinking_mean_steering_vectors_by_level.npy"))
visual_glm = torch.from_numpy(np.load("Data/Representation/Steering_vectors/SPLIT_steering_vectors_image_vs_no_image_glm.npy"))

cos_glm = torch.nn.functional.cosine_similarity(text_split_glm, visual_glm, dim=1)
print(cos_glm)

# %%
vals = cos_4b

x = np.arange(len(vals))

plt.figure(figsize=(8,4), dpi=300)
plt.plot(x, vals, marker='o', linewidth=2, color="#4C78A8")
plt.axhline(0, color="#666666", linewidth=1, linestyle="--")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Line Plot")
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
data = [
    ("Qwen3-VL-2B-Thinking", cos_2b),
    ("Qwen3-VL-4B-Thinking", cos_4b),
    ("Qwen3-VL-8B-Thinking", cos),
    ("GLM-4.1V-9B-Thinking", cos_glm),
]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300, sharex=False, sharey=False)

for ax, (name, vals) in zip(axs.ravel(), data):
    x = np.arange(len(vals))
    ax.plot(x, vals, marker='o', linewidth=3, color="#4C78A8", markersize=3)
    ax.axhline(0, color="#666666", linewidth=3, linestyle="--")
    ax.set_title(name, fontsize=24)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

for ax in axs.ravel():
    ax.set_xlabel("Layer", fontsize=20)
    ax.set_ylabel("Cosine Similarity", fontsize=20)


fig.tight_layout()
plt.show()


