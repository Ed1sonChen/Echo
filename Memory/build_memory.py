import os
import logging
import faiss
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip

DATASETS = [
    r"../Dataset/MMAD/MVTec-AD",
    r"../Dataset/MMAD/VisA"
]
REFERENCE_FILE = r"reference_image_locations.txt"
INDEX_SAVE_PATH = r"memory.index"
DEVICE = "cuda"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_data_location(base_dirs, output_file):
    """Traverse dataset directories and save train image paths."""
    with open(output_file, 'w') as f:
        for base_dir in base_dirs:
            for subfolder in os.listdir(base_dir):
                subfolder_path = os.path.join(base_dir, subfolder)
                train_folder = os.path.join(subfolder_path, "train")
                if os.path.isdir(subfolder_path) and os.path.exists(train_folder):
                    for root, _, files in os.walk(train_folder):
                        for file in files:
                            f.write(os.path.join(root, file) + "\n")
    logging.info(f"Image locations saved to {output_file}")


def build_memory(reference_file, index_save_path, device="cuda"):
    """Build FAISS memory index from reference images using CLIP embeddings."""
    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    # Read reference paths
    with open(reference_file, 'r') as file:
        image_paths = [line.strip() for line in file if line.strip()]
    logging.info(f"Total reference images: {len(image_paths)}")

    # Init FAISS index
    index_img = faiss.IndexHNSWFlat(512, 64, faiss.METRIC_INNER_PRODUCT)

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Embedding Images"):
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
            except Exception as e:
                logging.warning(f"Failed to process {image_path}: {e}")

    embeddings = np.vstack(embeddings)
    index_img.add(embeddings)

    logging.info(f"Total number of indexes: {index_img.ntotal}")

    # Save index
    faiss.write_index(index_img, index_save_path)
    logging.info(f"Index saved to {index_save_path}")


if __name__ == "__main__":
    logging.info("Step 1: Collecting image paths...")
    build_data_location(DATASETS, REFERENCE_FILE)

    logging.info("Step 2: Building FAISS memory index...")
    build_memory(REFERENCE_FILE, INDEX_SAVE_PATH, DEVICE)

    logging.info("All done!")
