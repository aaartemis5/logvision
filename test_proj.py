from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("Text projection type:", type(model.text_projection))
print("Visual projection type:", type(model.visual_projection))
