"""Diagnostic-only tools for Medium World; does not modify AttributeBindingViT.

- Oracle-matched related questions and counterfactual image pairs.
- Frozen per-object probes at three visual layers.
- Network Dissection-style patch/channel overlap with renderer masks.
- SpRAy-inspired spectral clustering of signed gradient×(image-background) maps.

The latter two are *adaptations*, not exact replications of their source papers.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

from src.medium_visual_world.learning import encode_query
from src.medium_visual_world.world import COLORS, SHAPES, Scene, from_dict, validate_scene, ANGLES
from src.medium_visual_world.render import render, BACKGROUND
from src.medium_visual_world.generate import HOLDOUT

TASK_QUERIES = {"color": "EXISTS_COLOR", "shape": "EXISTS_SHAPE",
                "conjunction": "HAS_COLOR_AND_SHAPE", "pair": "EXISTS_PAIR"}


def distinct_scenes(rows, limit=None, seed=19):
    scenes = {}
    for row in rows:
        scenes.setdefault(row["scene_id"], row)
    ids = sorted(scenes)
    random.Random(seed).shuffle(ids)
    selected = ids[:limit] if limit is not None else ids
    return [scenes[i] for i in selected]


def image_tensor(scene_dict):
    image = np.asarray(render(from_dict(scene_dict)).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).div_(255.0)


def oracle(scene_dict, task, color, shape):
    objs = scene_dict["objects"]
    colors, shapes = {o["color"] for o in objs}, {o["shape"] for o in objs}
    return int({"color": color in colors, "shape": shape in shapes,
                "conjunction": color in colors and shape in shapes,
                "pair": any(o["color"] == color and o["shape"] == shape for o in objs)}[task])


def query_tokens(task, color, shape):
    return [TASK_QUERIES[task]] + ([color] if task == "color" else
           [shape] if task == "shape" else [color, shape])


def same_scene_grid(rows, limit=None, seed=19, split="id_dev"):
    out = []
    for row in distinct_scenes(rows, limit, seed):
        scene = row["scene"]
        for color in COLORS:
            for shape in SHAPES:
                has_color = oracle(scene, "color", color, shape)
                has_shape = oracle(scene, "shape", color, shape)
                case = {(0,0):"both_absent",(0,1):"missing_color",
                        (1,0):"missing_shape",(1,1):"both_present"}[(has_color,has_shape)]
                for task in TASK_QUERIES:
                    target = color if task == "color" else shape if task == "shape" else f"{color}_{shape}"
                    out.append({"scene_id": row["scene_id"], "split": split,
                        "question_key": f"{row['scene_id']}|{task}|{target}",
                        "task": task, "color": color, "shape": shape,
                        "heldout_target": (color,shape) in HOLDOUT,
                        "logic_case":case,
                        "label": oracle(scene, task, color, shape), "scene": scene})
    return out


@torch.inference_mode()
def predict_rows(model, rows, device, batch_size=64, image_cache=None,
                 record_head_features=False):
    """Per-question probabilities; optional exact eight-feature head-input trace.

    A temporary pre-hook observes the *actual* input to the first head Linear.
    No duplicate/approximate forward pass and no permanent model modification.
    """
    model.eval()
    cache = {} if image_cache is None else image_cache
    output = []
    captured = []
    hook = None
    if record_head_features:
        if not isinstance(model.head[0], torch.nn.Linear) or model.head[0].in_features != 8:
            raise ValueError('Original AttributeBindingViT eight-feature head expected')
        hook = model.head[0].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy().copy()))
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start+batch_size]
            ims, queries = [], []
            for row in chunk:
                key = row.get('scene_key', row['scene_id'])
                if key not in cache:
                    cache[key] = image_tensor(row['scene'])
                ims.append(cache[key])
                queries.append(encode_query(query_tokens(row['task'], row['color'], row['shape'])))
            x = torch.stack(ims).to(device)
            q = torch.tensor(queries, dtype=torch.long, device=device)
            probs = model(x, q).softmax(-1)[:, 1].cpu().numpy()
            features = None
            if record_head_features:
                if len(captured) != 1:
                    raise AssertionError(f'Expected exactly one head-input capture, got {len(captured)}')
                features = captured.pop()
                if features.shape != (len(chunk), 8):
                    raise AssertionError(f'Unexpected head features: {features.shape}')
            for i, (row, prob) in enumerate(zip(chunk, probs)):
                record = {k: v for k, v in row.items() if k != 'scene'}
                clipped = float(np.clip(prob, 1e-7, 1.0-1e-7))
                nll = -(np.log(clipped) if row['label'] else np.log(1.0-clipped))
                record.update({'prob_yes': float(prob), 'prediction': int(prob >= 0.5),
                               'correct': int((prob >= 0.5) == bool(row['label'])),
                               'cross_entropy': float(nll)})
                if features is not None:
                    record.update({f'head_f{j}': float(features[i, j]) for j in range(8)})
                output.append(record)
    finally:
        if hook is not None:
            hook.remove()
    return pd.DataFrame(output)


def consistency_report(frame):
    keys = ["scene_id", "split", "color", "shape"]
    wide = (frame.pivot(index=keys, columns="task", values="prediction")
            .rename(columns=lambda task: "pred_" + task).reset_index())
    actual = (frame.pivot(index=keys, columns="task", values="correct")
              .rename(columns=lambda task: "correct_" + task).reset_index())
    wide = wide.merge(actual, on=keys, validate="one_to_one")
    wide["violates_binding_implies_attributes"] = (wide.pred_pair == 1) & (
        (wide.pred_color == 0) | (wide.pred_shape == 0))
    wide["violates_conjunction_definition"] = (
        wide.pred_conjunction != (wide.pred_color & wide.pred_shape))
    wide["missed_binding_despite_correct_prerequisites"] = (
        (wide.correct_color == 1) & (wide.correct_shape == 1) &
        (wide.correct_conjunction == 1) & (wide.correct_pair == 0))
    return wide


def _validate_changed(scene):
    validate_scene(from_dict(scene))
    return scene


def counterfactual_rows(rows, limit=None, seed=21, split="id_dev"):
    """Create valid No→Yes and invariant-label pairs; never invent oracle labels."""
    output = []
    for row in distinct_scenes(rows, limit, seed):
        scene = row["scene"]
        objects = scene["objects"]
        for color in COLORS:
            for shape in SHAPES:
                present_c = any(o["color"] == color for o in objects)
                present_s = any(o["shape"] == shape for o in objects)
                target = any(o["color"] == color and o["shape"] == shape for o in objects)
                edits = []
                if present_c and present_s and not target:
                    i = next(i for i,o in enumerate(objects) if o["color"] == color)
                    j = next(i for i,o in enumerate(objects) if o["shape"] == shape)
                    changed = deepcopy(scene)
                    changed["objects"][i]["color"], changed["objects"][j]["color"] = changed["objects"][j]["color"], changed["objects"][i]["color"]
                    edits.append(("binding_swap", "pair", changed))
                if present_c and not present_s:
                    # Keep rotation unchanged, avoiding a second intervention.
                    # Choose a color-matching object whose existing rotation
                    # remains valid for the target shape; otherwise skip.
                    possible = [i for i,o in enumerate(objects)
                                if o["color"] == color and o["angle"] in ANGLES[shape]]
                    if possible:
                        changed = deepcopy(scene)
                        changed["objects"][possible[0]]["shape"] = shape
                        edits.append(("shape_insert", "conjunction", changed))
                if present_s and not present_c:
                    changed = deepcopy(scene)
                    i = next(i for i,o in enumerate(objects) if o["shape"] == shape)
                    changed["objects"][i]["color"] = color
                    edits.append(("color_insert", "conjunction", changed))
                for edit, task, changed in edits:
                    _validate_changed(changed)
                    old_y = oracle(scene, task, color, shape)
                    new_y = oracle(changed, task, color, shape)
                    if (old_y, new_y) != (0, 1):
                        raise AssertionError((edit, old_y, new_y))
                    base = f"{row['scene_id']}_{color}_{shape}_{edit}"
                    for state, x_scene, y in [("before",scene,old_y),("after",changed,new_y)]:
                        output.append({"scene_id": row["scene_id"], "scene_key": base+state,
                            "pair_id": base, "state": state, "edit": edit, "split": split,
                            "task": task, "color": color, "shape": shape,
                            "heldout_target": (color,shape) in HOLDOUT,
                            "label": y, "scene": x_scene})
                    if edit == "binding_swap":
                        # The same edit must leave independent attribute presence unchanged.
                        for control_task in ("color", "shape", "conjunction"):
                            assert oracle(scene, control_task, color, shape) == oracle(changed, control_task, color, shape)
    if not output:
        raise ValueError(f"No valid counterfactual edits for {split}")
    return output


def counterfactual_report(predictions):
    cols = ["pair_id", "edit", "task", "split", "color", "shape"]
    x = predictions.pivot(index=cols, columns="state", values=["label", "prob_yes", "correct"])
    x.columns = ["_".join(z) for z in x.columns]
    x = x.reset_index()
    assert bool(((x.label_before == 0) & (x.label_after == 1)).all())
    x["delta_yes"] = x.prob_yes_after - x.prob_yes_before
    x["direction_correct"] = x.delta_yes > 0
    x["both_correct"] = (x.correct_before == 1) & (x.correct_after == 1)
    x["scene_id"] = x.pair_id.str.split("_", n=1).str[0]
    return x


def invariance_rows(counterfactual_examples):
    """Same oracle answer under an intervention; matched control questions."""
    grouped = {}
    for record in counterfactual_examples:
        grouped.setdefault(record["pair_id"], {})[record["state"]] = record
    controls = []
    for base_id, pair in grouped.items():
        before, after = pair["before"], pair["after"]
        edit = before["edit"]
        tasks = ("color", "shape", "conjunction") if edit == "binding_swap" else (
            ("color",) if edit == "shape_insert" else ("shape",))
        for task in tasks:
            base = f"{base_id}_invariant_{task}"
            old = oracle(before["scene"],task,before["color"],before["shape"])
            new = oracle(after["scene"],task,after["color"],after["shape"])
            if old != new:
                raise AssertionError("Control changed its oracle answer")
            for state,source in (("before",before),("after",after)):
                controls.append({**source,"pair_id":base,"state":state,
                                 "edit":f"{edit}_invariance_{task}",
                                 "task":task,"label":old})
    return controls


def invariance_report(predictions):
    cols=["pair_id","edit","task","split","color","shape"]
    x=predictions.pivot(index=cols,columns="state",values=["label","prob_yes","prediction","correct"])
    x.columns=["_".join(z) for z in x.columns]
    x=x.reset_index()
    assert (x.label_before==x.label_after).all()
    x["abs_delta_yes"]=(x.prob_yes_after-x.prob_yes_before).abs()
    x["prediction_flipped"]=x.prediction_before!=x.prediction_after
    x["both_correct"]=(x.correct_before==1)&(x.correct_after==1)
    x["scene_id"]=x.pair_id.str.split("_",n=1).str[0]
    return x


@torch.inference_mode()
def visual_stages(model, images):
    patches = model.patch(images).flatten(2).transpose(1, 2)
    encoded = model.encoder(patches + model.pos)
    fused = model.score_norm(model.pre_norm(patches) + 0.5*model.post_norm(encoded))
    return {"patch": patches, "encoder": encoded, "fused": fused}


def center_region(tokens, obj, patch_size=4, radius_pixels=8):
    grid = 64//patch_size
    x = (np.arange(grid)+0.5)*patch_size
    keep_x = np.abs(x - obj["x"]) <= radius_pixels
    keep_y = np.abs(x - obj["y"]) <= radius_pixels
    mask = torch.as_tensor(np.outer(keep_y, keep_x).reshape(-1), device=tokens.device)
    return tokens[:, mask].mean(1) if tokens.ndim == 3 else tokens[mask].mean(0)


@torch.inference_mode()
def extract_object_features(model, scene_rows, device, batch_size=32, patch_size=4):
    """Fixed square ROI around each oracle center; no shape-derived crop mask."""
    model.eval()
    output = {stage: [] for stage in ("patch", "encoder", "fused")}
    metadata = []
    for start in range(0, len(scene_rows), batch_size):
        batch = scene_rows[start:start+batch_size]
        tensor = torch.stack([image_tensor(r["scene"]) for r in batch]).to(device)
        stages = visual_stages(model, tensor)
        for i, row in enumerate(batch):
            for j, obj in enumerate(row["scene"]["objects"]):
                metadata.append({"scene_id": row["scene_id"], "object_index": j,
                                 "color": obj["color"], "shape": obj["shape"],
                                 "radius": obj["radius"], "angle": obj["angle"]})
                for name, encoded in stages.items():
                    output[name].append(center_region(encoded[i], obj, patch_size).cpu().numpy())
    return pd.DataFrame(metadata), {name: np.stack(items) for name,items in output.items()}


def fit_probe(train_x, train_y, eval_x, eval_y, seed=42):
    labels = np.unique(train_y)
    if len(labels) < 2:
        raise ValueError("Probe training requires at least two observed classes")
    unknown = set(np.unique(eval_y)) - set(labels)
    if unknown:
        raise ValueError(f"Probe evaluation contains unseen training classes: {unknown}")
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, random_state=seed))
    probe.fit(train_x, train_y)
    pred = probe.predict(eval_x)
    return {"accuracy": float(accuracy_score(eval_y, pred)),
            "macro_f1": float(f1_score(eval_y, pred, labels=labels,
                                      average="macro", zero_division=0)),
            "predictions": pred, "confusion": confusion_matrix(eval_y, pred, labels=labels),
            "class_order": labels, "probe": probe}


def foreground_concept_masks(scene_dict, patch_size=4, foreground_threshold=28):
    """Renderer-derived exact-object labels, converted to patch masks.

    Single-object rendering uses the same renderer/palette/anti-aliasing as the
    original image; threshold excludes near-background pixels but includes outlines.
    """
    masks = {f"color_{c}": np.zeros((64,64), bool) for c in COLORS}
    masks.update({f"shape_{s}": np.zeros((64,64), bool) for s in SHAPES})
    bg = np.asarray(BACKGROUND, dtype=np.int16)
    for obj in scene_dict["objects"]:
        solo = np.asarray(render(Scene((from_dict({"objects":[obj]}).objects[0],))), dtype=np.int16)
        mask = np.linalg.norm(solo - bg[None,None,:], axis=2) > foreground_threshold
        masks[f"color_{obj['color']}"] |= mask
        masks[f"shape_{obj['shape']}"] |= mask
    return {key: (F.avg_pool2d(torch.from_numpy(mask.astype(np.float32))[None,None],
                                 patch_size, patch_size)[0,0].numpy() >= 0.15)
            for key,mask in masks.items()}


@torch.inference_mode()
def collect_dissection(model, train_rows, dev_rows, device,
                      stages=("patch", "encoder", "fused"),
                      quantile=0.995, batch_size=24):
    """ViT channel/mask IoU with train-calibrated activation thresholds.

    `quantile` may be one float or a tuple of threshold quantiles. Reuse the
    exact same train and dev feature maps for a fair sensitivity comparison.
    """
    model.eval()
    quantiles = (quantile,) if isinstance(quantile, (int, float)) else tuple(quantile)
    if not quantiles or any(not 0.0 < q < 1.0 for q in quantiles):
        raise ValueError(f"Expected quantiles in (0,1), got {quantiles}")
    def feature_map(rows):
        ret = {stage: [] for stage in stages}
        for start in range(0, len(rows), batch_size):
            tensor = torch.stack([image_tensor(r["scene"]) for r in rows[start:start+batch_size]]).to(device)
            features = visual_stages(model, tensor)
            for stage in stages:
                ret[stage].append(features[stage].detach().cpu().numpy().transpose(0, 2, 1))
        return {stage: np.concatenate(ret[stage], axis=0) for stage in stages}
    calibrate = feature_map(train_rows)
    evaluated = feature_map(dev_rows)
    masks = [foreground_concept_masks(r["scene"]) for r in dev_rows]
    concepts = list(masks[0])
    targets = {concept: np.stack([mask[concept].reshape(-1) for mask in masks])
               for concept in concepts}
    results = []
    for stage in stages:
        train = calibrate[stage]
        for q in quantiles:
            thresholds = np.quantile(train, q, axis=(0, 2))
            active = evaluated[stage] >= thresholds[None, :, None]
            for concept, target in targets.items():
                intersection = np.logical_and(active, target[:, None, :]).sum((0, 2))
                union = np.logical_or(active, target[:, None, :]).sum((0, 2))
                for channel in range(active.shape[1]):
                    results.append({"stage": stage, "unit": channel, "concept": concept,
                        "iou": float(intersection[channel] / union[channel]) if union[channel] else 0.0,
                        "target_patches": int(target.sum()),
                        "activated_patches": int(active[:, channel].sum()),
                        "quantile": q, "threshold": float(thresholds[channel])})
    return pd.DataFrame(results)


def _background_tensor(device, dtype):
    return torch.tensor(BACKGROUND,device=device,dtype=dtype).view(1,3,1,1)/255


def signed_explanations(model, rows, device, batch_size=16, resolution=16):
    """Signed gradient×(input−background) of YES logit margin (NOT LRP)."""
    model.eval()
    result, meta = [], []
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        x=torch.stack([image_tensor(r["scene"]) for r in batch]).to(device)
        x.requires_grad_(True)
        q=torch.tensor([encode_query(query_tokens(r["task"],r["color"],r["shape"]))
                        for r in batch],dtype=torch.long,device=device)
        # One forward pass only; use the YES-vs-NO margin for all samples.
        logits=model(x,q)
        margin=logits[:,1]-logits[:,0]
        gradients=torch.autograd.grad(margin.sum(), x)[0]
        signed=(gradients*(x-_background_tensor(device,x.dtype))).sum(1,keepdim=True)
        resized=F.adaptive_avg_pool2d(signed,(resolution,resolution))[:,0]
        maps=resized.detach().cpu().numpy()
        maps=maps/(np.abs(maps).sum(axis=(1,2),keepdims=True)+1e-12)
        result.append(maps)
        probabilities=logits.softmax(-1)[:,1].detach().cpu().numpy()
        for r,p in zip(batch, probabilities):
            meta.append({k:v for k,v in r.items() if k!="scene"} | {
                "prob_yes":float(p),"prediction":int(p>=.5),"correct":int((p>=.5)==bool(r["label"]))})
    return np.concatenate(result),pd.DataFrame(meta)


def spray_cluster(maps, n_clusters=3, neighbors=12, n_eigen=12, seed=42):
    """Exploratory SpRAy-style graph/eigenspectrum on explanation maps.

    kNN affinities use RBF weights of cosine distance. Unlike original SpRAy,
    explanation maps here come from gradient×(image-background), not LRP.
    """
    n=len(maps)
    if n<=max(neighbors,n_clusters):
        raise ValueError("Need more maps than neighbors/clusters")
    vectors=maps.reshape(n,-1)
    norms=np.linalg.norm(vectors,axis=1)
    if (norms<1e-10).any():
        raise ValueError("Zero explanation maps; inspect before clustering")
    nearest = NearestNeighbors(n_neighbors=neighbors + 1, metric="cosine").fit(vectors)
    distances, indices = nearest.kneighbors(vectors)
    # A tied zero-distance neighbor can appear before the query itself.
    # Remove the *actual* self-index instead of assuming column 0 is self.
    neighbors_by_row = [
        [(float(distance), int(index)) for distance, index in zip(row_dist, row_idx)
         if int(index) != row_number][:neighbors]
        for row_number, (row_dist, row_idx) in enumerate(zip(distances, indices))
    ]
    nonzero_distances = [distance for row in neighbors_by_row for distance, _ in row
                         if distance > 1e-8]
    scale = max(float(np.median(nonzero_distances)), 1e-4) if nonzero_distances else 1e-4
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i, row in enumerate(neighbors_by_row):
        for distance, j in row:
            adjacency[i, j] = np.exp(-(distance / scale) ** 2)
    adjacency=np.maximum(adjacency,adjacency.T)
    np.fill_diagonal(adjacency,1.0)
    L=laplacian(adjacency,normed=True)
    eigenvalues=eigh(L,subset_by_index=(0,min(n-1,n_eigen-1)),eigvals_only=True)
    components = connected_components(adjacency, directed=False, return_labels=False)
    if components > 1:
        # Spectral components can be informative, but fixed-K clustering of a
        # disconnected affinity graph is not a stable taxonomy.
        print(f"WARNING: SpRAy affinity graph has {components} connected components")
    clusterer=SpectralClustering(n_clusters=n_clusters,affinity="precomputed",
                                assign_labels="kmeans",random_state=seed,n_init=20)
    labels=clusterer.fit_predict(adjacency)
    gaps=np.diff(eigenvalues)
    return labels,eigenvalues,gaps,adjacency


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_hash_audit(data_dir, expected=None):
    """Hash frozen manifests; reject accidental dataset replacement."""
    data_dir = Path(data_dir)
    files = ("train.jsonl", "id_dev.jsonl", "ood_dev.jsonl")
    hashes = {name: sha256_file(data_dir / name) for name in files}
    if expected is not None and hashes != expected:
        raise AssertionError(f"Frozen dataset changed: {hashes} != {expected}")
    return hashes


def unique_question_rows(frame):
    """Remove repeated Color/Shape queries from a per-(color,shape) logic grid.

    Retain the complete 4-task Cartesian grid separately for logical checks.
    """
    if "question_key" not in frame.columns:
        raise KeyError("Missing question_key; cannot deduplicate joint grid")
    scope = [key for key in ("schedule", "seed", "step", "split") if key in frame.columns]
    key_columns = scope + ["question_key"]
    # Vectorized consistency check: the complete five-seed grid can contain
    # >1 million unique queries. Iterating through each group is too slow.
    counts = frame.groupby(key_columns, sort=False)[["prediction", "label"]].nunique()
    invalid = (counts["prediction"] > 1) | (counts["label"] > 1)
    if invalid.any():
        raise AssertionError(f"Duplicated questions disagree: {counts[invalid].head(4)}")
    unique = frame.drop_duplicates(key_columns).copy()
    # The requested *pair* and combined logical case do not exist as
    # meaningful metadata for Color-only or Shape-only tasks. Otherwise a
    # random first duplicated row assigns arbitrary heldout/case metadata.
    irrelevant = unique["task"].isin(("color", "shape"))
    unique["heldout_target"] = unique["heldout_target"].astype(object)
    unique.loc[irrelevant, "heldout_target"] = None
    unique.loc[irrelevant, "logic_case"] = None
    return unique


def make_explanation_questions(dataset_rows, scene_limit=160, seed=80, split="id_dev"):
    """One fixed Binding query per scene, with explicit selection criteria.

    ID-dev: balanced positives versus hard negatives where both attributes
    exist but are attached to different objects. OOD-dev: retain each scene's
    original held-out Binding query, stratified by its oracle label.

    Explanations are not interchangeable across these different query sets.
    """
    if scene_limit % 2:
        raise ValueError("Expected an even number of explanation scenes")
    rng = random.Random(seed + 901)
    selected = distinct_scenes(dataset_rows, limit=None, seed=seed)

    if split == "ood_dev":
        by_label = {0: [], 1: []}
        for row in selected:
            if row["template"] != "pair":
                raise ValueError("OOD explanation questions require Pair records")
            color, shape = row["question"]["args"]
            label = oracle(row["scene"], "pair", color, shape)
            if label != int(row["label"]):
                raise AssertionError("OOD record disagrees with oracle")
            if (color, shape) not in HOLDOUT:
                raise AssertionError("OOD target is not a held-out pair")
            by_label[label].append({
                "scene_id": row["scene_id"], "scene": row["scene"],
                "task": "pair", "split": split, "color": color,
                "shape": shape, "heldout_target": True,
                "logic_case": "present_pair" if label else "attributes_separate",
                "label": label,
            })
        half = scene_limit // 2
        if any(len(by_label[label]) < half for label in (0, 1)):
            raise ValueError("Insufficient balanced OOD scenes")
        questions = [item for label in (0, 1) for item in by_label[label][:half]]
    else:
        available = []
        for row in selected:
            objs = row["scene"]["objects"]
            colors = {obj["color"] for obj in objs}
            shapes = {obj["shape"] for obj in objs}
            positives = sorted({(obj["color"], obj["shape"]) for obj in objs})
            negatives = sorted((c, s) for c in colors for s in shapes
                               if (c, s) not in positives and (c, s) not in HOLDOUT)
            if positives and negatives:
                available.append((row, positives, negatives))
        if len(available) < scene_limit:
            raise ValueError("Too few ID scenes with hard Binding negatives")
        questions = []
        for i, (row, positives, negatives) in enumerate(available[:scene_limit]):
            label = i % 2
            color, shape = rng.choice(positives if label else negatives)
            if oracle(row["scene"], "pair", color, shape) != label:
                raise AssertionError("Generated Pair query disagrees with oracle")
            questions.append({
                "scene_id": row["scene_id"], "scene": row["scene"],
                "task": "pair", "split": split, "color": color,
                "shape": shape, "heldout_target": (color, shape) in HOLDOUT,
                "logic_case": "present_pair" if label else "attributes_separate",
                "label": label,
            })

    if len(questions) != scene_limit:
        raise AssertionError("Unexpected explanation question count")
    if len({q["scene_id"] for q in questions}) != scene_limit:
        raise AssertionError("Explanation questions must use distinct scenes")
    if sum(q["label"] for q in questions) != scene_limit // 2:
        raise AssertionError("Explanation labels must be balanced")
    rng.shuffle(questions)
    return questions

# Refined frozen-feature Shape diagnostics. The ViT weights remain unchanged.
# The two readouts use the same spatial samples, so their difference isolates
# the effect of discarding patch arrangement (subject to decoder capacity).

def recolor_probe_scenes(scene_rows, seed=20261110):
    """Recolor training scenes without changing objects' shapes or geometry.

    A uniformly shuffled sequence of global palette shifts keeps the colors
    within each scene distinct wherever they originally were. All Shape×Color
    combinations become available for fitting the diagnostic decoder, not ViT.
    """
    indices = list(range(len(scene_rows)))
    random.Random(seed).shuffle(indices)
    shifts = [0] * len(scene_rows)
    for position, index in enumerate(indices):
        shifts[index] = position % len(COLORS)
    out = []
    for row, shift in zip(scene_rows, shifts):
        changed = deepcopy(row)
        for obj in changed["scene"]["objects"]:
            original = COLORS.index(obj["color"])
            obj["color"] = COLORS[(original + shift) % len(COLORS)]
        changed["source_scene_id"] = row["scene_id"]
        changed["scene_id"] = from_dict(changed["scene"]).scene_id()
        out.append(changed)
    return out


def _local_grid(tokens, obj, patch_size=4, crop_tokens=5):
    """Bilinearly sample a fixed local H×W token grid centered on an object.

    The sampling grid depends only on the oracle center, never on Shape,
    Color, object silhouette, or the target label. Preserves spatial order.
    """
    if crop_tokens % 2 != 1:
        raise ValueError("crop_tokens must be odd")
    side = 64 // patch_size
    channels = tokens.shape[-1]
    feature_map = tokens.reshape(side, side, channels).permute(2, 0, 1)[None]
    offsets = torch.arange(crop_tokens, device=tokens.device, dtype=tokens.dtype)
    offsets = (offsets - crop_tokens // 2) * patch_size
    ys = obj["y"] + offsets
    xs = obj["x"] + offsets
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((2 * grid_x / 64 - 1, 2 * grid_y / 64 - 1), dim=-1)[None]
    sampled = F.grid_sample(feature_map, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
    return sampled[0].permute(1, 2, 0).contiguous()


@torch.inference_mode()
def extract_spatial_object_features(model, scene_rows, device, batch_size=24,
                                    patch_size=4, crop_tokens=5, image_cache=None):
    """Return matched spatial (25d) and mean-pooled (d) frozen Shape features."""
    model.eval()
    features = {stage: {"spatial": [], "mean": []}
                for stage in ("patch", "encoder", "fused")}
    metadata = []
    cache = {} if image_cache is None else image_cache
    for start in range(0, len(scene_rows), batch_size):
        group = scene_rows[start:start + batch_size]
        batch_images = []
        for row in group:
            key = row['scene_id']
            if key not in cache:
                cache[key] = image_tensor(row['scene'])
            batch_images.append(cache[key])
        stages = visual_stages(model, torch.stack(batch_images).to(device))
        for index, row in enumerate(group):
            for object_index, obj in enumerate(row["scene"]["objects"]):
                metadata.append({
                    "scene_id": row["scene_id"],
                    "source_scene_id": row.get("source_scene_id", row["scene_id"]),
                    "object_index": object_index,
                    "shape": obj["shape"],
                    "color": obj["color"],
                    "radius": obj["radius"],
                    "angle": obj["angle"],
                })
                for stage, tokens in stages.items():
                    grid = _local_grid(tokens[index], obj, patch_size, crop_tokens)
                    features[stage]["spatial"].append(grid.flatten().cpu().numpy())
                    features[stage]["mean"].append(grid.mean((0, 1)).cpu().numpy())
    return pd.DataFrame(metadata), {
        stage: {kind: np.asarray(items, dtype=np.float32)
                for kind, items in choices.items()}
        for stage, choices in features.items()
    }


def shape_color_weights(metadata):
    """Equal total training weight for every Shape×Color cell (20 cells)."""
    cells = metadata.groupby(["shape", "color"]).size()
    required = pd.MultiIndex.from_product([SHAPES, COLORS], names=["shape", "color"])
    observed = cells.reindex(required, fill_value=0)
    if (observed == 0).any():
        raise ValueError("Recolored probe training is missing Shape×Color cells")
    per_object = 1.0 / np.asarray([
        cells.loc[(shape, color)]
        for shape, color in zip(metadata["shape"], metadata["color"])
    ], dtype=np.float64)
    per_object *= len(per_object) / per_object.sum()
    balance = observed.rename("n_objects").reset_index()
    balance["total_sample_weight"] = 1.0 / len(required)
    return per_object, balance


def fit_spatial_shape_decoder(features, labels, sample_weight=None):
    """One linear multinomial probe, shared across ID and OOD evaluations.

    Ridge/LSQR avoids expensive covariance inversion for the 25×48 spatial
    input. It is linear in the supplied features and uses no dev-label tuning.
    """
    from sklearn.linear_model import RidgeClassifier
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    estimator = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=10.0, solver="lsqr", tol=1e-5)
    )
    estimator.fit(x, y, ridgeclassifier__sample_weight=sample_weight)
    return estimator


def evaluate_shape_decoder(probe, features, truth):
    predictions = probe.predict(features)
    order = np.asarray(SHAPES)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=order,
                                    average="macro", zero_division=0)),
        "predictions": predictions,
    }


def extract_silhouette_features(scene_rows, crop_pixels=24,
                                foreground_threshold=28):
    """Raw-image geometry sanity check, independent of learned visual tokens.

    Fixed square crop around oracle center; foreground is extracted directly
    from the rendered RGB image without inspecting the true Shape label.
    """
    if crop_pixels % 2:
        raise ValueError("crop_pixels must be even")
    bg = np.asarray(BACKGROUND, dtype=np.float32)
    rows, vectors = [], []
    for row in scene_rows:
        pixels = np.asarray(render(from_dict(row["scene"])), dtype=np.float32)
        foreground = (np.linalg.norm(pixels - bg[None, None], axis=2)
                      > foreground_threshold).astype(np.float32)
        for j, obj in enumerate(row["scene"]["objects"]):
            x0, y0 = obj["x"] - crop_pixels // 2, obj["y"] - crop_pixels // 2
            padded = np.pad(foreground, crop_pixels, mode="constant")
            crop = padded[y0 + crop_pixels:y0 + crop_pixels * 2,
                          x0 + crop_pixels:x0 + crop_pixels * 2]
            vectors.append(crop.reshape(-1))
            rows.append({"scene_id": row["scene_id"], "object_index": j,
                         "shape": obj["shape"], "color": obj["color"],
                         "radius": obj["radius"], "angle": obj["angle"]})
    return pd.DataFrame(rows), np.asarray(vectors, dtype=np.float32)
