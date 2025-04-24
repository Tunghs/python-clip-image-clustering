import os
import shutil
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import umap.umap_ as umap

def get_all_image_paths(root_folder):
    image_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def load_and_embed_images(image_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    embeddings = []
    valid_paths = []

    for path in tqdm(image_paths, desc="Embedding images"):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy().flatten()
                embeddings.append(embedding)
                valid_paths.append(path)
        except Exception as e:
            print(f"Error with {path}: {e}")

    return np.array(embeddings), valid_paths

# 클러스터 수 자동 추정
def find_optimal_clusters(embeddings, min_k=2, max_k=10):
    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"K={k}, Silhouette Score={score:.4f}")
        if score > best_score:
            best_k = k
            best_score = score

    print(f"최적 클러스터 수: {best_k} (Silhouette Score: {best_score:.4f})")
    return best_k

def cluster_and_save(embeddings, paths, output_folder, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    for idx, label in enumerate(labels):
        cluster_folder = os.path.join(output_folder, f"cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.copy(paths[idx], os.path.join(cluster_folder, os.path.basename(paths[idx])))

    print("클러스터링 및 저장 완료!")
    return labels

def visualize_clusters(embeddings, labels):
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', s=30)
    plt.title("UMAP Clustering Visualization")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(image_root_dir, output_dir, min_k=2, max_k=10):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = get_all_image_paths(image_root_dir)
    embeddings, valid_paths = load_and_embed_images(image_paths)

    # 클러스터 수 자동 추정
    optimal_k = find_optimal_clusters(embeddings, min_k=min_k, max_k=max_k)

    # 클러스터링 및 저장
    labels = cluster_and_save(embeddings, valid_paths, output_dir, n_clusters=optimal_k)

    # 시각화
    visualize_clusters(embeddings, labels)

input_image_dir = r""
outut_dir = r"";

# 사용 예
main(input_image_dir, outut_dir, min_k=2, max_k=16)