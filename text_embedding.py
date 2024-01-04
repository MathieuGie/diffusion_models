import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = ["Un petit ourson trop mignon"]
text_input = clip.tokenize(text).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)
