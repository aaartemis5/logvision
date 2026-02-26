from transformers import CLIPProcessor, CLIPModel
import torch

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

c_in = processor(text=["a photo of a black garment"], return_tensors="pt", padding=True)
try:
    outputs = model(**c_in)
    print("Direct forward works! text_embeds:", outputs.text_embeds.shape)
except Exception as e:
    print("Direct forward failed:", e)

# What about get_text_features? Let's just project manually if it's the wrong shape.
c_feat = model.get_text_features(**c_in)
if isinstance(c_feat, torch.Tensor):
    if c_feat.shape[-1] != 512 and hasattr(model, "text_projection"):
        print("Tensor returned but shape is not 512! Projecting...")
        if isinstance(model.text_projection, torch.nn.Linear):
            c_feat = model.text_projection(c_feat)
        else:
            c_feat = c_feat @ model.text_projection.weight
    print("Final shape:", c_feat.shape)
