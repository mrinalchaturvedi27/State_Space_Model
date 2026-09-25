import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
from huggingface_hub import login
from pose_format import Pose
import wandb
from torch.amp import autocast, GradScaler
import cv2
from typing import Dict, List
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
set_seed()

# ===================== Pose Visualization (128×128) =====================
def draw_skeleton(img, points, connections, color=(0, 255, 0), thickness=1):
    h, w = img.shape[:2]
    for start, end in connections:
        if 0 <= start < len(points) and 0 <= end < len(points):
            pt1 = (int(points[start][0]), int(points[start][1]))
            pt2 = (int(points[end][0]), int(points[end][1]))
            if all(0 <= c < max(h, w) for c in pt1 + pt2):
                cv2.line(img, pt1, pt2, color, thickness)
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

def create_pose_image(pose_features: np.ndarray, frame_idx: int) -> Image.Image:
    """Compact 128×128 skeleton. 152 keypoints are fully retained at this resolution."""
    height, width = 128, 128
    img_np = np.zeros((height, width, 3), dtype=np.uint8) + 240

    try:
        scale = 52.0
        offset = 64.0
        hands_l = pose_features[:42].reshape(-1, 2) * scale + offset
        hands_r = pose_features[42:84].reshape(-1, 2) * scale + offset
        body = pose_features[84:150].reshape(-1, 2) * scale + offset
        face = pose_features[150:].reshape(-1, 2) * scale + offset if len(pose_features) > 150 else np.zeros((10, 2))

        body_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
        hand_connections = [(0, 1), (1, 2), (2, 3), (3, 4)] * 5

        draw_skeleton(img_np, body, body_connections, color=(255, 100, 0))
        draw_skeleton(img_np, hands_l, hand_connections, color=(0, 200, 255))
        draw_skeleton(img_np, hands_r, hand_connections, color=(0, 200, 255))
        draw_skeleton(img_np, face[:16], [(0, 1), (2, 3)], color=(100, 100, 255))
        cv2.putText(img_np, f"{frame_idx}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    except Exception:
        points = pose_features.reshape(-1, 2) * 52 + 64
        for i, (x, y) in enumerate(points[:40]):
            if 0 < x < width and 0 < y < height:
                cv2.circle(img_np, (int(x), int(y)), 2, (0, 0, 255), -1)

    return Image.fromarray(img_np)

# ===================== Pose feature extraction =====================
def find_nearest_unmasked_frame(data, confidence_measure, start_idx):
    for offset in range(1, len(data)):
        forward_idx = start_idx + offset
        backward_idx = start_idx - offset
        if forward_idx < len(data):
            if hasattr(data, 'mask') and data.mask is not None:
                if not np.any(data.mask[forward_idx]) and np.min(confidence_measure[forward_idx]) > 0.8:
                    return forward_idx
            elif np.min(confidence_measure[forward_idx]) > 0.8:
                return forward_idx
        if backward_idx >= 0:
            if hasattr(data, 'mask') and data.mask is not None:
                if not np.any(data.mask[backward_idx]) and np.min(confidence_measure[backward_idx]) > 0.8:
                    return backward_idx
            elif np.min(confidence_measure[backward_idx]) > 0.8:
                return backward_idx
    return start_idx

def sample_unmasked_frames(data, confidence_measure, step=4):
    sampled_frames, confidence_frames = [], []
    for i in range(0, len(data), step):
        if i >= len(data):
            break
        if hasattr(data, 'mask') and data.mask is not None and np.any(data.mask[i]):
            nearest_idx = find_nearest_unmasked_frame(data, confidence_measure, i)
            sampled_frames.append(data[nearest_idx])
            confidence_frames.append(confidence_measure[nearest_idx])
        else:
            sampled_frames.append(data[i])
            confidence_frames.append(confidence_measure[i])
    return (np.stack(sampled_frames) if sampled_frames else np.array([]),
            np.stack(confidence_frames) if confidence_frames else np.array([]))

def get_pose_keypoints(pose, component, step_frames=4):
    try:
        available_components = list(pose.header.components.keys()) if isinstance(pose.header.components, dict) else \
            [comp.name for comp in pose.header.components if hasattr(comp, 'name')]
        if component not in available_components:
            return np.array([])
        component_header = pose.header.components[component] if isinstance(pose.header.components, dict) else \
            next((c for c in pose.header.components if hasattr(c, 'name') and c.name == component), None)
        if component_header is None:
            return np.array([])
        num_keypoints = len(component_header.points)
        start_idx = sum(len(pose.header.components[comp].points) if isinstance(pose.header.components, dict) else
                        len(next(c for c in pose.header.components if c.name == comp).points)
                        for comp in available_components if comp < component)
        end_idx = start_idx + num_keypoints
        data = np.squeeze(pose.body.data[:, :, start_idx:end_idx, :], axis=1)
        confidence = np.squeeze(pose.body.confidence[:, :, start_idx:end_idx], axis=1)

        if component == 'POSE_LANDMARKS':
            pose.normalize(pose.header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                                          p2=("POSE_LANDMARKS", "LEFT_SHOULDER")))
        elif component == 'FACE_LANDMARKS':
            pose.normalize(pose.header.normalization_info(p1=("FACE_LANDMARKS", "48"),
                                                          p2=("FACE_LANDMARKS", "278")))
        elif component in ['LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS']:
            pose.normalize(pose.header.normalization_info(p1=(component, "INDEX_FINGER_MCP"),
                                                          p2=(component, "PINKY_MCP")))

        keypoints, _ = sample_unmasked_frames(data[:, :, :2], confidence, step=step_frames)
        if len(keypoints) == 0:
            return np.array([])
        min_vals = np.min(keypoints, axis=(0, 1))
        max_vals = np.max(keypoints, axis=(0, 1))
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        keypoints = 2 * (keypoints - min_vals) / range_vals - 1
        return keypoints.reshape((keypoints.shape[0], -1))
    except Exception as e:
        logger.error(f"Error processing {component}: {e}")
        return np.array([])

def extract_features_from_pose(pose_path, step_frames=4):
    try:
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())
        hand_left = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', step_frames)
        hand_right = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', step_frames)
        body_pose = get_pose_keypoints(pose, 'POSE_LANDMARKS', step_frames)
        face = get_pose_keypoints(pose, 'FACE_LANDMARKS', step_frames)

        if hand_left.size == 0: hand_left = np.zeros((1, 42))
        if hand_right.size == 0: hand_right = np.zeros((1, 42))
        if body_pose.size == 0: body_pose = np.zeros((1, 66))
        if face.size == 0: face = np.zeros((1, 936))

        min_frames = min(h.shape[0] if h.size > 0 else float('inf') for h in [hand_left, hand_right, body_pose, face])
        if min_frames == float('inf'):
            return np.zeros((1, 1086))

        hand_left = hand_left[:min_frames] if hand_left.size > 0 else np.zeros((min_frames, 42))
        hand_right = hand_right[:min_frames] if hand_right.size > 0 else np.zeros((min_frames, 42))
        body_pose = body_pose[:min_frames] if body_pose.size > 0 else np.zeros((min_frames, 66))
        face = face[:min_frames] if face.size > 0 else np.zeros((min_frames, 936))

        return np.concatenate((hand_left, hand_right, body_pose, face), axis=1)
    except Exception as e:
        logger.error(f"Error extracting features from {pose_path}: {e}")
        return np.zeros((1, 1086))

# ===================== Dataset =====================
class QwenPoseDataset(Dataset):
    def __init__(self, video_data: Dict, processor, max_frames=8, video_dir=None,
                 add_noise=False, step_frames=4):
        self.video_ids = list(video_data.keys())
        self.translations = list(video_data.values())
        self.processor = processor
        self.max_frames = max_frames
        self.video_dir = video_dir
        self.add_noise = add_noise
        self.step_frames = step_frames
        self.feature_cache = {}

    def __len__(self):
        return len(self.video_ids)

    def load_pose_features(self, video_id):
        try:
            video_path = os.path.join(self.video_dir, f'{video_id}.pose')
            if not os.path.exists(video_path):
                return np.zeros((self.max_frames, 1086))
            pose_features = extract_features_from_pose(video_path, self.step_frames)
            if pose_features.shape[0] < self.max_frames:
                padded = np.zeros((self.max_frames, 1086))
                padded[:pose_features.shape[0]] = pose_features
                pose_features = padded
            else:
                pose_features = pose_features[:self.max_frames]
            if self.add_noise and random.random() < 0.4:
                pose_features = pose_features + np.random.normal(0, 0.015, pose_features.shape)
            return pose_features
        except Exception as e:
            logger.error(f"Pose load error {video_id}: {e}")
            return np.zeros((self.max_frames, 1086))

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        translation = self.translations[idx]

        pose_features = self.feature_cache.get(video_id)
        if pose_features is None:
            pose_features = self.load_pose_features(video_id)
            if len(self.feature_cache) < 4000:
                self.feature_cache[video_id] = pose_features

        frame_images: List[Image.Image] = [
            create_pose_image(pose_features[f_idx], f_idx) for f_idx in range(self.max_frames)
        ]

        system_prompt = (
            "You are an expert sign language interpreter. "
            "Translate the given sequence of pose frames into clear, natural English."
        )

        user_content = [{"type": "image", "image": img} for img in frame_images]
        user_content.append({"type": "text", "text": "Translate this sign language video to English."})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": translation}
        ]

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_inputs = self.processor(
            text=[full_text],
            images=frame_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        prompt_messages = messages[:-1]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=frame_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_input_ids": prompt_inputs["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_inputs["attention_mask"].squeeze(0),
            "video_id": video_id,
            "reference": translation,
        }

        if "pixel_values" in full_inputs:
            sample["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
            sample["prompt_pixel_values"] = prompt_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs:
            sample["image_grid_thw"] = full_inputs["image_grid_thw"]
            sample["prompt_image_grid_thw"] = prompt_inputs["image_grid_thw"]

        return sample

def collate_fn(batch):
    if not batch:
        return {}

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["prompt_input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    prompt_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["prompt_attention_mask"] for b in batch], batch_first=True, padding_value=0
    )

    pixel_values = torch.stack([b["pixel_values"] for b in batch]) if "pixel_values" in batch[0] else None
    prompt_pixel_values = torch.stack([b["prompt_pixel_values"] for b in batch]) if "prompt_pixel_values" in batch[0] else None
    image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch]) if "image_grid_thw" in batch[0] else None
    prompt_image_grid_thw = torch.cat([b["prompt_image_grid_thw"] for b in batch]) if "prompt_image_grid_thw" in batch[0] else None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "pixel_values": pixel_values,
        "prompt_pixel_values": prompt_pixel_values,
        "image_grid_thw": image_grid_thw,
        "prompt_image_grid_thw": prompt_image_grid_thw,
        "video_ids": [b["video_id"] for b in batch],
        "references": [b["reference"] for b in batch],
    }

# ===================== Metrics =====================
def compute_metrics(predictions, references):
    if not predictions or not references:
        return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0, 'ROUGE-L': 0, 'SacreBLEU': 0}
    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu1 = bleu2 = bleu3 = bleu4 = rouge_l = sacre = 0.0
    valid = 0
    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            continue
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        bleu1 += sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 += sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 += sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu4 += sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        rouge_l += rouge.score(ref, pred)['rougeL'].fmeasure
        sacre += sacrebleu.sentence_bleu(pred, [ref]).score
        valid += 1
    if valid == 0:
        return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0, 'ROUGE-L': 0, 'SacreBLEU': 0}
    return {
        'BLEU-1': bleu1 / valid, 'BLEU-2': bleu2 / valid,
        'BLEU-3': bleu3 / valid, 'BLEU-4': bleu4 / valid,
        'ROUGE-L': rouge_l / valid, 'SacreBLEU': sacre / valid
    }

@torch.no_grad()
def evaluate_predictions(model, dataloader, processor, device, max_new_tokens=48, output_csv=None):
    model.eval()
    predictions, references, video_ids = [], [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        gen_kwargs = {
            "input_ids": batch["prompt_input_ids"],
            "attention_mask": batch["prompt_attention_mask"],
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,                 # reduced for speed
            "do_sample": False,             # greedy for faster eval
            "no_repeat_ngram_size": 3,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        if batch.get("prompt_pixel_values") is not None:
            gen_kwargs["pixel_values"] = batch["prompt_pixel_values"]
        if batch.get("prompt_image_grid_thw") is not None:
            gen_kwargs["image_grid_thw"] = batch["prompt_image_grid_thw"]

        generated_ids = model.generate(**gen_kwargs)

        for gen_ids, ref, vid in zip(generated_ids, batch["references"], batch["video_ids"]):
            prompt_len = batch["prompt_input_ids"].shape[1]
            new_tokens = gen_ids[prompt_len:] if gen_ids.shape[0] > prompt_len else gen_ids
            pred_text = processor.batch_decode([new_tokens], skip_special_tokens=True)[0].strip()
            pred_text = pred_text.split("\n")[0].strip()
            predictions.append(pred_text)
            references.append(ref)
            video_ids.append(vid)

    if output_csv:
        pd.DataFrame({
            "video_id": video_ids,
            "prediction": predictions,
            "ground_truth": references
        }).to_csv(output_csv, index=False)

    return compute_metrics(predictions, references)

# ===================== Main =====================
def main(gpu_id: int = 0, eval_every_n: int = 5):
    hyperparameters = {
        'project_name': 'EACL-qwen3-vl-8b-pose-128x8',
        'dataset_name': 'isign1.0',
        'train_csv': '/DATA405/sanjeet/Projects/ARR-August-26/pose/csv_files/train_split_unicode_filtered.csv',
        'val_csv': '/DATA405/sanjeet/Projects/ARR-August-26/pose/csv_files/val_split_unicode_filtered.csv',
        'test_csv': '/DATA405/sanjeet/Projects/ARR-August-26/pose/csv_files/test_split_unicode_filtered.csv',
        'pose_dir': '/DATA7/vaibhav/isign/Data/iSign-poses_v1.1/',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
        'max_frames': 16,                      # critical for speed & memory
        'batch_size': 4,                      # now feasible with 128×128 + 8 frames
        'epochs': 100,
        'learning_rate': 6e-6,
        'warmup_steps': 300,
        'gradient_accumulation_steps': 8,     # effective batch = 16
        'weight_decay': 0.01,
        'num_workers': 4,
        'max_new_tokens': 48,
        'early_stopping_patience': 25,
        'gpu_id': gpu_id,
        'use_lora': True,
        'eval_every_n': eval_every_n
    }

    base_dir = os.path.join(
        '/DATA405/sanjeet/Projects/ARR-August-26/pose/iSign/qwen-3-8b-visual-tokens',
        hyperparameters['project_name'], hyperparameters['dataset_name']
    )
    for d in ['output/checkpoints', 'output/best_model', 'output/results']:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    wandb.init(project=hyperparameters['project_name'], config=hyperparameters)

    device = torch.device(f"cuda:{hyperparameters['gpu_id']}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger.info(f"Using device: {device}")

    hf_token = os.environ.get("HF_TOKEN")
    login(token=hf_token)

    def load_data(csv_path):
        df = pd.read_csv(csv_path)
        # df = df[:100]
        return {vid: trans for vid, trans in zip(df['uid'], df['text'])}

    train_data = load_data(hyperparameters['train_csv'])
    val_data = load_data(hyperparameters['val_csv'])
    test_data = load_data(hyperparameters['test_csv'])
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Strongly constrain visual token count
    processor = AutoProcessor.from_pretrained(
        hyperparameters['model_name'],
        trust_remote_code=True,
        # min_pixels=64 * 28 * 28,
        # max_pixels=256 * 28 * 28
        min_pixels=32*28*28,
        max_pixels=128*28*28
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hyperparameters['model_name'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(device)

    # Gradient checkpointing – large memory saving for the 8B model
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    if hyperparameters['use_lora']:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,                           # slightly lower rank for speed
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = QwenPoseDataset(
        train_data, processor,
        max_frames=hyperparameters['max_frames'],
        video_dir=hyperparameters['pose_dir'],
        add_noise=True
    )
    val_dataset = QwenPoseDataset(
        val_data, processor,
        max_frames=hyperparameters['max_frames'],
        video_dir=hyperparameters['pose_dir'],
        add_noise=False
    )
    test_dataset = QwenPoseDataset(
        test_data, processor,
        max_frames=hyperparameters['max_frames'],
        video_dir=hyperparameters['pose_dir'],
        add_noise=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=hyperparameters['num_workers'],
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=hyperparameters['num_workers'],
        pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=hyperparameters['num_workers'],
        pin_memory=True, persistent_workers=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay']
    )
    total_steps = len(train_loader) * hyperparameters['epochs'] // hyperparameters['gradient_accumulation_steps']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hyperparameters['warmup_steps'],
        num_training_steps=max(total_steps, 1)
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = []

    logger.info("Starting optimized training: 128×128 images, 8 frames, batch=2, LoRA r=16, gradient checkpointing")

    for epoch in range(hyperparameters['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparameters['epochs']}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }
                if batch.get("pixel_values") is not None:
                    model_inputs["pixel_values"] = batch["pixel_values"]
                if batch.get("image_grid_thw") is not None:
                    model_inputs["image_grid_thw"] = batch["image_grid_thw"]

                outputs = model(**model_inputs)
                loss = outputs.loss / hyperparameters['gradient_accumulation_steps']

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * hyperparameters['gradient_accumulation_steps']

            if (step + 1) % hyperparameters['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item() * hyperparameters['gradient_accumulation_steps']:.4f}"})

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1})

        # Validation loss (cheap)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                    model_inputs = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "labels": batch["labels"],
                    }
                    if batch.get("pixel_values") is not None:
                        model_inputs["pixel_values"] = batch["pixel_values"]
                    if batch.get("image_grid_thw") is not None:
                        model_inputs["image_grid_thw"] = batch["image_grid_thw"]
                    outputs = model(**model_inputs)
                    val_loss += outputs.loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
        wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})

        # Full generation evaluation only every eval_every_n epochs
        if (epoch + 1) % hyperparameters['eval_every_n'] == 0 or epoch == 0 or epoch >= hyperparameters['epochs'] - 3:
            val_csv = os.path.join(base_dir, 'output/results', f'val_predictions_epoch_{epoch+1}.csv')
            test_csv = os.path.join(base_dir, 'output/results', f'test_predictions_epoch_{epoch+1}.csv')

            val_metrics = evaluate_predictions(model, val_loader, processor, device, output_csv=val_csv)
            test_metrics = evaluate_predictions(model, test_loader, processor, device, output_csv=test_csv)

            logger.info(f"Val  BLEU-4: {val_metrics['BLEU-4']:.4f} | ROUGE-L: {val_metrics['ROUGE-L']:.4f}")
            logger.info(f"Test BLEU-4: {test_metrics['BLEU-4']:.4f} | ROUGE-L: {test_metrics['ROUGE-L']:.4f}")

            metrics_history.append({
                'epoch': epoch + 1,
                'val_loss': round(avg_val_loss, 4),
                **{f'val_{k}': round(v, 4) for k, v in val_metrics.items()},
                **{f'test_{k}': round(v, 4) for k, v in test_metrics.items()}
            })
            pd.DataFrame(metrics_history).to_excel(
                os.path.join(base_dir, 'output/results', 'metrics_summary.xlsx'), index=False
            )
            wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()},
                       **{f"test/{k}": v for k, v in test_metrics.items()},
                       "epoch": epoch + 1})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(os.path.join(base_dir, 'output/best_model'))
            processor.save_pretrained(os.path.join(base_dir, 'output/best_model'))
            patience_counter = 0
            logger.info(f"New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= hyperparameters['early_stopping_patience']:
            logger.info("Early stopping triggered.")
            break

    model.save_pretrained(os.path.join(base_dir, 'output'))
    processor.save_pretrained(os.path.join(base_dir, 'output'))
    wandb.finish()
    logger.info("Training finished successfully.")

if __name__ == "__main__":
    import sys
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    eval_n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    main(gpu_id=gpu, eval_every_n=eval_n)

# import os
# import torch
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import logging
# from torch.utils.data import Dataset, DataLoader
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
# from huggingface_hub import login
# from pose_format import Pose
# import wandb
# from torch.amp import autocast, GradScaler
# import cv2
# from typing import Dict, List
# import sacrebleu
# from rouge_score import rouge_scorer
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# import random
# from PIL import Image

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     random.seed(seed)
# set_seed()

# # ===================== Pose Visualization (down-sampled to 224×224) =====================
# def draw_skeleton(img, points, connections, color=(0, 255, 0), thickness=2):
#     h, w = img.shape[:2]
#     for start, end in connections:
#         if 0 <= start < len(points) and 0 <= end < len(points):
#             pt1 = (int(points[start][0]), int(points[start][1]))
#             pt2 = (int(points[end][0]), int(points[end][1]))
#             if all(0 <= c < max(h, w) for c in pt1 + pt2):
#                 cv2.line(img, pt1, pt2, color, thickness)
#     for x, y in points:
#         if 0 <= x < w and 0 <= y < h:
#             cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

# def create_pose_image(pose_features: np.ndarray, frame_idx: int) -> Image.Image:
#     """Render a compact 224×224 skeleton image. All kinematic information is retained."""
#     height, width = 224, 224
#     img_np = np.zeros((height, width, 3), dtype=np.uint8) + 240

#     try:
#         # Scale normalized keypoints [-1,1] → pixel coordinates centered on the canvas
#         scale = 90.0
#         offset = 112.0
#         hands_l = pose_features[:42].reshape(-1, 2) * scale + offset
#         hands_r = pose_features[42:84].reshape(-1, 2) * scale + offset
#         body = pose_features[84:150].reshape(-1, 2) * scale + offset
#         face = pose_features[150:].reshape(-1, 2) * scale + offset if len(pose_features) > 150 else np.zeros((10, 2))

#         body_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
#         hand_connections = [(0, 1), (1, 2), (2, 3), (3, 4)] * 5

#         draw_skeleton(img_np, body, body_connections, color=(255, 100, 0))
#         draw_skeleton(img_np, hands_l, hand_connections, color=(0, 200, 255))
#         draw_skeleton(img_np, hands_r, hand_connections, color=(0, 200, 255))
#         draw_skeleton(img_np, face[:20], [(0, 1), (2, 3)], color=(100, 100, 255))
#         cv2.putText(img_np, f"F{frame_idx}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#     except Exception:
#         points = pose_features.reshape(-1, 2) * 90 + 112
#         for i, (x, y) in enumerate(points[:40]):
#             if 0 < x < width and 0 < y < height:
#                 cv2.circle(img_np, (int(x), int(y)), 3, (0, 0, 255), -1)

#     return Image.fromarray(img_np)

# # ===================== Pose feature extraction =====================
# def find_nearest_unmasked_frame(data, confidence_measure, start_idx):
#     for offset in range(1, len(data)):
#         forward_idx = start_idx + offset
#         backward_idx = start_idx - offset
#         if forward_idx < len(data):
#             if hasattr(data, 'mask') and data.mask is not None:
#                 if not np.any(data.mask[forward_idx]) and np.min(confidence_measure[forward_idx]) > 0.8:
#                     return forward_idx
#             elif np.min(confidence_measure[forward_idx]) > 0.8:
#                 return forward_idx
#         if backward_idx >= 0:
#             if hasattr(data, 'mask') and data.mask is not None:
#                 if not np.any(data.mask[backward_idx]) and np.min(confidence_measure[backward_idx]) > 0.8:
#                     return backward_idx
#             elif np.min(confidence_measure[backward_idx]) > 0.8:
#                 return backward_idx
#     return start_idx

# def sample_unmasked_frames(data, confidence_measure, step=4):
#     sampled_frames, confidence_frames = [], []
#     for i in range(0, len(data), step):
#         if i >= len(data):
#             break
#         if hasattr(data, 'mask') and data.mask is not None and np.any(data.mask[i]):
#             nearest_idx = find_nearest_unmasked_frame(data, confidence_measure, i)
#             sampled_frames.append(data[nearest_idx])
#             confidence_frames.append(confidence_measure[nearest_idx])
#         else:
#             sampled_frames.append(data[i])
#             confidence_frames.append(confidence_measure[i])
#     return (np.stack(sampled_frames) if sampled_frames else np.array([]),
#             np.stack(confidence_frames) if confidence_frames else np.array([]))

# def get_pose_keypoints(pose, component, step_frames=4):
#     try:
#         available_components = list(pose.header.components.keys()) if isinstance(pose.header.components, dict) else \
#             [comp.name for comp in pose.header.components if hasattr(comp, 'name')]
#         if component not in available_components:
#             return np.array([])
#         component_header = pose.header.components[component] if isinstance(pose.header.components, dict) else \
#             next((c for c in pose.header.components if hasattr(c, 'name') and c.name == component), None)
#         if component_header is None:
#             return np.array([])
#         num_keypoints = len(component_header.points)
#         start_idx = sum(len(pose.header.components[comp].points) if isinstance(pose.header.components, dict) else
#                         len(next(c for c in pose.header.components if c.name == comp).points)
#                         for comp in available_components if comp < component)
#         end_idx = start_idx + num_keypoints
#         data = np.squeeze(pose.body.data[:, :, start_idx:end_idx, :], axis=1)
#         confidence = np.squeeze(pose.body.confidence[:, :, start_idx:end_idx], axis=1)

#         if component == 'POSE_LANDMARKS':
#             pose.normalize(pose.header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
#                                                           p2=("POSE_LANDMARKS", "LEFT_SHOULDER")))
#         elif component == 'FACE_LANDMARKS':
#             pose.normalize(pose.header.normalization_info(p1=("FACE_LANDMARKS", "48"),
#                                                           p2=("FACE_LANDMARKS", "278")))
#         elif component in ['LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS']:
#             pose.normalize(pose.header.normalization_info(p1=(component, "INDEX_FINGER_MCP"),
#                                                           p2=(component, "PINKY_MCP")))

#         keypoints, _ = sample_unmasked_frames(data[:, :, :2], confidence, step=step_frames)
#         if len(keypoints) == 0:
#             return np.array([])
#         min_vals = np.min(keypoints, axis=(0, 1))
#         max_vals = np.max(keypoints, axis=(0, 1))
#         range_vals = max_vals - min_vals
#         range_vals[range_vals == 0] = 1.0
#         keypoints = 2 * (keypoints - min_vals) / range_vals - 1
#         return keypoints.reshape((keypoints.shape[0], -1))
#     except Exception as e:
#         logger.error(f"Error processing {component}: {e}")
#         return np.array([])

# def extract_features_from_pose(pose_path, step_frames=4):
#     try:
#         with open(pose_path, "rb") as f:
#             pose = Pose.read(f.read())
#         hand_left = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', step_frames)
#         hand_right = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', step_frames)
#         body_pose = get_pose_keypoints(pose, 'POSE_LANDMARKS', step_frames)
#         face = get_pose_keypoints(pose, 'FACE_LANDMARKS', step_frames)

#         if hand_left.size == 0: hand_left = np.zeros((1, 42))
#         if hand_right.size == 0: hand_right = np.zeros((1, 42))
#         if body_pose.size == 0: body_pose = np.zeros((1, 66))
#         if face.size == 0: face = np.zeros((1, 936))

#         min_frames = min(h.shape[0] if h.size > 0 else float('inf') for h in [hand_left, hand_right, body_pose, face])
#         if min_frames == float('inf'):
#             return np.zeros((1, 1086))

#         hand_left = hand_left[:min_frames] if hand_left.size > 0 else np.zeros((min_frames, 42))
#         hand_right = hand_right[:min_frames] if hand_right.size > 0 else np.zeros((min_frames, 42))
#         body_pose = body_pose[:min_frames] if body_pose.size > 0 else np.zeros((min_frames, 66))
#         face = face[:min_frames] if face.size > 0 else np.zeros((min_frames, 936))

#         return np.concatenate((hand_left, hand_right, body_pose, face), axis=1)
#     except Exception as e:
#         logger.error(f"Error extracting features from {pose_path}: {e}")
#         return np.zeros((1, 1086))

# # ===================== Dataset =====================
# class QwenPoseDataset(Dataset):
#     def __init__(self, video_data: Dict, processor, max_frames=16, video_dir=None,
#                  add_noise=False, step_frames=4):
#         self.video_ids = list(video_data.keys())
#         self.translations = list(video_data.values())
#         self.processor = processor
#         self.max_frames = max_frames
#         self.video_dir = video_dir
#         self.add_noise = add_noise
#         self.step_frames = step_frames
#         self.feature_cache = {}

#     def __len__(self):
#         return len(self.video_ids)

#     def load_pose_features(self, video_id):
#         try:
#             video_path = os.path.join(self.video_dir, f'{video_id}.pose')
#             if not os.path.exists(video_path):
#                 return np.zeros((self.max_frames, 1086))
#             pose_features = extract_features_from_pose(video_path, self.step_frames)
#             if pose_features.shape[0] < self.max_frames:
#                 padded = np.zeros((self.max_frames, 1086))
#                 padded[:pose_features.shape[0]] = pose_features
#                 pose_features = padded
#             else:
#                 pose_features = pose_features[:self.max_frames]
#             if self.add_noise and random.random() < 0.5:
#                 pose_features = pose_features + np.random.normal(0, 0.02, pose_features.shape)
#             return pose_features
#         except Exception as e:
#             logger.error(f"Pose load error {video_id}: {e}")
#             return np.zeros((self.max_frames, 1086))

#     def __getitem__(self, idx):
#         video_id = self.video_ids[idx]
#         translation = self.translations[idx]

#         pose_features = self.feature_cache.get(video_id)
#         if pose_features is None:
#             pose_features = self.load_pose_features(video_id)
#             if len(self.feature_cache) < 3000:
#                 self.feature_cache[video_id] = pose_features

#         # Exactly max_frames compact 224×224 images
#         frame_images: List[Image.Image] = []
#         for f_idx in range(self.max_frames):
#             frame_images.append(create_pose_image(pose_features[f_idx], f_idx))

#         system_prompt = (
#             "You are an expert sign language interpreter. "
#             "Analyze the sequence of pose frames and translate into fluent, natural English."
#         )

#         user_content = [{"type": "image", "image": img} for img in frame_images]
#         user_content.append({"type": "text", "text": "Translate this sign language video to English."})

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_content},
#             {"role": "assistant", "content": translation}
#         ]

#         # Full sequence (training)
#         full_text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=False
#         )
#         full_inputs = self.processor(
#             text=[full_text],
#             images=frame_images,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=4096
#         )

#         # Prompt-only sequence (generation)
#         prompt_messages = messages[:-1]
#         prompt_text = self.processor.apply_chat_template(
#             prompt_messages, tokenize=False, add_generation_prompt=True
#         )
#         prompt_inputs = self.processor(
#             text=[prompt_text],
#             images=frame_images,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=4096
#         )

#         input_ids = full_inputs["input_ids"].squeeze(0)
#         attention_mask = full_inputs["attention_mask"].squeeze(0)
#         labels = input_ids.clone()

#         prompt_len = prompt_inputs["input_ids"].shape[1]
#         labels[:prompt_len] = -100

#         sample = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels,
#             "prompt_input_ids": prompt_inputs["input_ids"].squeeze(0),
#             "prompt_attention_mask": prompt_inputs["attention_mask"].squeeze(0),
#             "video_id": video_id,
#             "reference": translation,
#         }

#         if "pixel_values" in full_inputs:
#             sample["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
#             sample["prompt_pixel_values"] = prompt_inputs["pixel_values"].squeeze(0)
#         if "image_grid_thw" in full_inputs:
#             sample["image_grid_thw"] = full_inputs["image_grid_thw"]
#             sample["prompt_image_grid_thw"] = prompt_inputs["image_grid_thw"]

#         return sample

# def collate_fn(batch):
#     if not batch:
#         return {}

#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         [b["input_ids"] for b in batch], batch_first=True, padding_value=0
#     )
#     attention_mask = torch.nn.utils.rnn.pad_sequence(
#         [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
#     )
#     labels = torch.nn.utils.rnn.pad_sequence(
#         [b["labels"] for b in batch], batch_first=True, padding_value=-100
#     )
#     prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
#         [b["prompt_input_ids"] for b in batch], batch_first=True, padding_value=0
#     )
#     prompt_attention_mask = torch.nn.utils.rnn.pad_sequence(
#         [b["prompt_attention_mask"] for b in batch], batch_first=True, padding_value=0
#     )

#     pixel_values = torch.stack([b["pixel_values"] for b in batch]) if "pixel_values" in batch[0] else None
#     prompt_pixel_values = torch.stack([b["prompt_pixel_values"] for b in batch]) if "prompt_pixel_values" in batch[0] else None
#     image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch]) if "image_grid_thw" in batch[0] else None
#     prompt_image_grid_thw = torch.cat([b["prompt_image_grid_thw"] for b in batch]) if "prompt_image_grid_thw" in batch[0] else None

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#         "prompt_input_ids": prompt_input_ids,
#         "prompt_attention_mask": prompt_attention_mask,
#         "pixel_values": pixel_values,
#         "prompt_pixel_values": prompt_pixel_values,
#         "image_grid_thw": image_grid_thw,
#         "prompt_image_grid_thw": prompt_image_grid_thw,
#         "video_ids": [b["video_id"] for b in batch],
#         "references": [b["reference"] for b in batch],
#     }

# # ===================== Metrics =====================
# def compute_metrics(predictions, references):
#     if not predictions or not references:
#         return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0, 'ROUGE-L': 0, 'SacreBLEU': 0}
#     smoothing = SmoothingFunction().method1
#     rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     bleu1 = bleu2 = bleu3 = bleu4 = rouge_l = sacre = 0.0
#     valid = 0
#     for pred, ref in zip(predictions, references):
#         if not pred.strip() or not ref.strip():
#             continue
#         ref_tokens = [ref.split()]
#         pred_tokens = pred.split()
#         bleu1 += sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
#         bleu2 += sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
#         bleu3 += sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
#         bleu4 += sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
#         rouge_l += rouge.score(ref, pred)['rougeL'].fmeasure
#         sacre += sacrebleu.sentence_bleu(pred, [ref]).score
#         valid += 1
#     if valid == 0:
#         return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0, 'ROUGE-L': 0, 'SacreBLEU': 0}
#     return {
#         'BLEU-1': bleu1 / valid, 'BLEU-2': bleu2 / valid,
#         'BLEU-3': bleu3 / valid, 'BLEU-4': bleu4 / valid,
#         'ROUGE-L': rouge_l / valid, 'SacreBLEU': sacre / valid
#     }

# @torch.no_grad()
# def evaluate_predictions(model, dataloader, processor, device, max_new_tokens=64, output_csv=None):
#     model.eval()
#     predictions, references, video_ids = [], [], []
#     for batch in tqdm(dataloader, desc="Evaluating"):
#         batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

#         gen_kwargs = {
#             "input_ids": batch["prompt_input_ids"],
#             "attention_mask": batch["prompt_attention_mask"],
#             "max_new_tokens": max_new_tokens,
#             "num_beams": 4,
#             "do_sample": True,
#             "temperature": 0.7,
#             "top_p": 0.9,
#             "no_repeat_ngram_size": 3,
#             "pad_token_id": processor.tokenizer.pad_token_id,
#             "eos_token_id": processor.tokenizer.eos_token_id,
#         }
#         if batch.get("prompt_pixel_values") is not None:
#             gen_kwargs["pixel_values"] = batch["prompt_pixel_values"]
#         if batch.get("prompt_image_grid_thw") is not None:
#             gen_kwargs["image_grid_thw"] = batch["prompt_image_grid_thw"]

#         generated_ids = model.generate(**gen_kwargs)

#         for gen_ids, ref, vid in zip(generated_ids, batch["references"], batch["video_ids"]):
#             prompt_len = batch["prompt_input_ids"].shape[1]
#             new_tokens = gen_ids[prompt_len:] if gen_ids.shape[0] > prompt_len else gen_ids
#             pred_text = processor.batch_decode([new_tokens], skip_special_tokens=True)[0].strip()
#             pred_text = pred_text.split("\n")[0].strip()
#             predictions.append(pred_text)
#             references.append(ref)
#             video_ids.append(vid)

#     if output_csv:
#         pd.DataFrame({
#             "video_id": video_ids,
#             "prediction": predictions,
#             "ground_truth": references
#         }).to_csv(output_csv, index=False)

#     return compute_metrics(predictions, references)

# # ===================== Main =====================
# def main(gpu_id: int = 0, eval_every_n: int = 3):
#     hyperparameters = {
#         'project_name': 'EACL-qwen-3-8b-vl-2b-pose-optimized-224',
#         'dataset_name': 'isign1.0',
#         'train_csv': '/DATA405/sanjeet/Projects/ARR-August-26/pose/csv_files/train_split_unicode_filtered.csv',
#         'val_csv': '/DATA405/sanjeet/Projects/ARR-August-26/pose/csv_files/val_split_unicode_filtered.csv',
#         'test_csv': '/DATA405/sanjeet/Projects/ARR-August-26/pose/csv_files/test_split_unicode_filtered.csv',
#         'pose_dir': '/DATA7/vaibhav/isign/Data/iSign-poses_v1.1/',
#         'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
#         'max_frames': 16,                     # Optimal temporal resolution
#         'batch_size': 4,                      # Fits 80 GB A100 with room to spare
#         'epochs': 300,
#         'learning_rate': 8e-6,
#         'warmup_steps': 400,
#         'gradient_accumulation_steps': 4,     # Effective batch = 32
#         'weight_decay': 0.01,
#         'num_workers': 4,
#         'max_new_tokens': 64,
#         'early_stopping_patience': 12,
#         'gpu_id': gpu_id,
#         'use_lora': True,
#         'eval_every_n': eval_every_n
#     }

#     base_dir = os.path.join(
#         '/DATA405/sanjeet/Projects/ARR-August-26/pose/iSign/qwen-3-8b-visual-tokens',
#         hyperparameters['project_name'], hyperparameters['dataset_name']
#     )
#     for d in ['output/checkpoints', 'output/best_model', 'output/results']:
#         os.makedirs(os.path.join(base_dir, d), exist_ok=True)

#     wandb.init(project=hyperparameters['project_name'], config=hyperparameters)

#     device = torch.device(f"cuda:{hyperparameters['gpu_id']}" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.is_available():
#         torch.cuda.set_device(device)
#     logger.info(f"Using device: {device}")

#     # Prefer environment variable; fall back to the token present in the original codebase
#     hf_token = os.environ.get("HF_TOKEN")
#     login(token=hf_token)

#     def load_data(csv_path):
#         df = pd.read_csv(csv_path)
#         return {vid: trans for vid, trans in zip(df['uid'], df['text'])}

#     train_data = load_data(hyperparameters['train_csv'])
#     val_data = load_data(hyperparameters['val_csv'])
#     test_data = load_data(hyperparameters['test_csv'])
#     logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

#     processor = AutoProcessor.from_pretrained(
#         hyperparameters['model_name'],
#         trust_remote_code=True,
#         min_pixels=256 * 28 * 28,   # encourage lower visual token count
#         max_pixels=1280 * 28 * 28
#     )
#     model = Qwen3VLForConditionalGeneration.from_pretrained(
#         hyperparameters['model_name'],
#         torch_dtype=torch.bfloat16,
#         attn_implementation="sdpa",
#         trust_remote_code=True
#     ).to(device)

#     if hyperparameters['use_lora']:
#         from peft import LoraConfig, get_peft_model
#         lora_config = LoraConfig(
#             r=32,
#             lora_alpha=64,
#             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )
#         model = get_peft_model(model, lora_config)
#         model.print_trainable_parameters()

#     train_dataset = QwenPoseDataset(
#         train_data, processor,
#         max_frames=hyperparameters['max_frames'],
#         video_dir=hyperparameters['pose_dir'],
#         add_noise=True
#     )
#     val_dataset = QwenPoseDataset(
#         val_data, processor,
#         max_frames=hyperparameters['max_frames'],
#         video_dir=hyperparameters['pose_dir'],
#         add_noise=False
#     )
#     test_dataset = QwenPoseDataset(
#         test_data, processor,
#         max_frames=hyperparameters['max_frames'],
#         video_dir=hyperparameters['pose_dir'],
#         add_noise=False
#     )

#     train_loader = DataLoader(
#         train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True,
#         collate_fn=collate_fn, num_workers=hyperparameters['num_workers'], pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False,
#         collate_fn=collate_fn, num_workers=hyperparameters['num_workers'], pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False,
#         collate_fn=collate_fn, num_workers=hyperparameters['num_workers'], pin_memory=True
#     )

#     optimizer = torch.optim.AdamW(
#         [p for p in model.parameters() if p.requires_grad],
#         lr=hyperparameters['learning_rate'],
#         weight_decay=hyperparameters['weight_decay']
#     )
#     total_steps = len(train_loader) * hyperparameters['epochs'] // hyperparameters['gradient_accumulation_steps']
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=hyperparameters['warmup_steps'],
#         num_training_steps=max(total_steps, 1)
#     )
#     scaler = GradScaler(enabled=torch.cuda.is_available())

#     best_val_loss = float('inf')
#     patience_counter = 0
#     metrics_history = []

#     logger.info("Starting optimized training (224×224 pose images, max_frames=16, batch=8)")

#     for epoch in range(hyperparameters['epochs']):
#         model.train()
#         epoch_loss = 0.0
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparameters['epochs']}")

#         for step, batch in enumerate(progress_bar):
#             batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

#             with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
#                 model_inputs = {
#                     "input_ids": batch["input_ids"],
#                     "attention_mask": batch["attention_mask"],
#                     "labels": batch["labels"],
#                 }
#                 if batch.get("pixel_values") is not None:
#                     model_inputs["pixel_values"] = batch["pixel_values"]
#                 if batch.get("image_grid_thw") is not None:
#                     model_inputs["image_grid_thw"] = batch["image_grid_thw"]

#                 outputs = model(**model_inputs)
#                 loss = outputs.loss / hyperparameters['gradient_accumulation_steps']

#             scaler.scale(loss).backward()
#             epoch_loss += loss.item() * hyperparameters['gradient_accumulation_steps']

#             if (step + 1) % hyperparameters['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_loader):
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             progress_bar.set_postfix({"loss": f"{loss.item() * hyperparameters['gradient_accumulation_steps']:.4f}"})

#         avg_train_loss = epoch_loss / max(len(train_loader), 1)
#         logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
#         wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1})

#         # Validation loss
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validation"):
#                 batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#                 with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
#                     model_inputs = {
#                         "input_ids": batch["input_ids"],
#                         "attention_mask": batch["attention_mask"],
#                         "labels": batch["labels"],
#                     }
#                     if batch.get("pixel_values") is not None:
#                         model_inputs["pixel_values"] = batch["pixel_values"]
#                     if batch.get("image_grid_thw") is not None:
#                         model_inputs["image_grid_thw"] = batch["image_grid_thw"]
#                     outputs = model(**model_inputs)
#                     val_loss += outputs.loss.item()
#         avg_val_loss = val_loss / max(len(val_loader), 1)
#         logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
#         wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})

#         # Periodic full evaluation
#         if (epoch + 1) % hyperparameters['eval_every_n'] == 0 or epoch == 0 or epoch >= hyperparameters['epochs'] - 5:
#             val_csv = os.path.join(base_dir, 'output/results', f'val_predictions_epoch_{epoch+1}.csv')
#             test_csv = os.path.join(base_dir, 'output/results', f'test_predictions_epoch_{epoch+1}.csv')

#             val_metrics = evaluate_predictions(model, val_loader, processor, device, output_csv=val_csv)
#             test_metrics = evaluate_predictions(model, test_loader, processor, device, output_csv=test_csv)

#             logger.info(f"Val  BLEU-4: {val_metrics['BLEU-4']:.4f} | ROUGE-L: {val_metrics['ROUGE-L']:.4f}")
#             logger.info(f"Test BLEU-4: {test_metrics['BLEU-4']:.4f} | ROUGE-L: {test_metrics['ROUGE-L']:.4f}")

#             metrics_history.append({
#                 'epoch': epoch + 1,
#                 'val_loss': round(avg_val_loss, 4),
#                 **{f'val_{k}': round(v, 4) for k, v in val_metrics.items()},
#                 **{f'test_{k}': round(v, 4) for k, v in test_metrics.items()}
#             })
#             pd.DataFrame(metrics_history).to_excel(
#                 os.path.join(base_dir, 'output/results', 'metrics_summary.xlsx'), index=False
#             )
#             wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()},
#                        **{f"test/{k}": v for k, v in test_metrics.items()},
#                        "epoch": epoch + 1})

#         # Checkpointing
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             model.save_pretrained(os.path.join(base_dir, 'output/best_model'))
#             processor.save_pretrained(os.path.join(base_dir, 'output/best_model'))
#             patience_counter = 0
#             logger.info(f"New best model saved (val_loss={best_val_loss:.4f})")
#         else:
#             patience_counter += 1

#         if patience_counter >= hyperparameters['early_stopping_patience']:
#             logger.info("Early stopping triggered.")
#             break

#     model.save_pretrained(os.path.join(base_dir, 'output'))
#     processor.save_pretrained(os.path.join(base_dir, 'output'))
#     wandb.finish()
#     logger.info("Training finished successfully.")

# if __name__ == "__main__":
#     import sys
#     gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 3
#     eval_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
#     main(gpu_id=gpu, eval_every_n=eval_n)