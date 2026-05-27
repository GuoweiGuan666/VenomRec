import torch
import numpy as np


def forward_inference(model, image_input, text_input):
    """调用预训练的VIP5模型进行前向推理，返回图像与文本的Embedding。"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if isinstance(image_input, np.ndarray):
        image_input = torch.from_numpy(image_input)
    if isinstance(text_input, np.ndarray):
        text_input = torch.from_numpy(text_input)

    device = next(model.parameters()).device
    image_input = image_input.to(device)
    text_input = text_input.to(device)

    with torch.no_grad():
        image_embedding = model.encoder.visual_embedding(image_input)
        text_embedding = model.encoder.embed_tokens(text_input)

    return {
        "image_embedding": image_embedding,
        "text_embedding": text_embedding,
    }


def compute_and_cache_embeddings(model, dataset_items):
    """对每个数据集的Targeted Item和Popular Item计算并缓存Embedding。"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cache = {}
    for dataset_name, items in dataset_items.items():
        targeted = items.get("targeted_item", {})
        popular = items.get("popular_item", {})

        targeted_emb = (
            forward_inference(
                model,
                targeted.get("image_input"),
                targeted.get("text_input"),
            )
            if targeted
            else {}
        )

        popular_emb = (
            forward_inference(
                model,
                popular.get("image_input"),
                popular.get("text_input"),
            )
            if popular
            else {}
        )

        cache[dataset_name] = {
            "targeted_item": targeted_emb,
            "popular_item": popular_emb,
        }

    return cache
