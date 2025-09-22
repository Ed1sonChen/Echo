import os
import json
import csv
import random
import logging
from collections import defaultdict

import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import clip
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from expert_generator import expert_generator
from transformers import Qwen2VLForConditionalGeneration

# ==========================
# Configurations
# ==========================
CONFIG = {
    "seed": 123,
    "device": "cuda",
    "clip_model": "ViT-B/16",
    "qwen_path": "Qwen/Qwen2-VL-7B-Instruct",
    "reference_index": r"../Memory/memory.index",
    "reference_images": r"../Memory/reference_image_locations.txt",
    "annotation_file": r"../Annotation/DS-MVTec.json",
    "domain_knowledge": r"../Knowledge Guide/domain_knowledge_detection.json",
    "results_csv": r"Results_Qwen2VL.csv"
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_seed(seed=123):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_image_feature(image_path, clip_model, preprocess, device="cuda"):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().squeeze()

def find_all_descriptions(json_file_path, img_path):
    """Retrieve domain knowledge description for an object from JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        object_name = img_path.split('/')[1]

        for _, sub_dict in data.items():
            if isinstance(sub_dict, dict) and object_name in sub_dict:
                return {"object_name": object_name, "descriptions": sub_dict[object_name]}

        return {"object_name": object_name, "descriptions": "No descriptions found."}

    except FileNotFoundError:
        return {"error": f"File not found: {json_file_path}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON."}

def load_reference_images(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def evaluate_model():
    set_seed(CONFIG["seed"])

    # Load CLIP
    clip_model, preprocess = clip.load(CONFIG["clip_model"], device=CONFIG["device"])

    # Load FAISS index
    index_img = faiss.read_index(CONFIG["reference_index"])
    image_paths = load_reference_images(CONFIG["reference_images"])

    # Load Qwen model & processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        CONFIG["qwen_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(CONFIG["device"])
    processor = AutoProcessor.from_pretrained(CONFIG["qwen_path"], trust_remote_code=True, use_fast=False)

    # Load dataset
    with open(CONFIG["annotation_file"], 'r') as f:
        data = json.load(f)

    metrics = defaultdict(lambda: {'y_true': [], 'y_pred': []})

    with open(CONFIG["results_csv"], 'w', newline='') as csvfile:
        fieldnames = ['Image Path', 'Question', 'Predicted Answer', 'Correct Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (img_path, item_value) in enumerate(data.items()):
            logging.info(f"Processing item {idx + 1} of {len(data)}")

            query_image_path = f"../Dataset/MMAD/{img_path}"
            query_image_feature = get_image_feature(query_image_path, clip_model, preprocess)

            # Find most similar image
            D, I = index_img.search(np.expand_dims(query_image_feature, axis=0), k=1)
            reference_image_path = image_paths[I[0][0]]

            # Domain knowledge
            domain_knowledge = find_all_descriptions(CONFIG["domain_knowledge"], img_path)

            # Loop through conversations
            for conversation in item_value['conversation']:
                question = conversation['Question']
                correct_answer = conversation['Answer']
                options = conversation['Options']
                question_type = conversation['type']

                options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])

                messages = expert_generator(reference_image_path, query_image_path, question_type, question, options_text, domain_knowledge)

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(CONFIG["device"])

                response_ids = model.generate(**inputs, max_length=5000)
                response_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, response_ids)]
                response = processor.batch_decode(response_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predicted_answer = response[0] if response else "N/A"

                metrics[question_type]['y_true'].append(correct_answer)
                metrics[question_type]['y_pred'].append(predicted_answer)

                writer.writerow({
                    'Image Path': img_path,
                    'Question': question,
                    'Predicted Answer': predicted_answer,
                    'Correct Answer': correct_answer
                })

    # Report metrics
    for question_type, values in metrics.items():
        y_true = values['y_true']
        y_pred = values['y_pred']
        y_true_filtered = [y for y, p in zip(y_true, y_pred) if p != "N/A"]
        y_pred_filtered = [p[0] if isinstance(p, list) else p for p in y_pred if p != "N/A"]

        accuracy = accuracy_score(y_true_filtered, y_pred_filtered) if y_true_filtered else 'N/A'
        logging.info(f"\nQuestion Type: {question_type}\nAccuracy: {accuracy}")


if __name__ == "__main__":
    evaluate_model()
