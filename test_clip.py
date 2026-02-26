from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

c_in = processor(text=["a photo of a black garment"], return_tensors="pt", padding=True)
c_feat = model.get_text_features(**c_in)

print("Type of c_feat:", type(c_feat))
if isinstance(c_feat, torch.Tensor):
    print("Shape of c_feat:", c_feat.shape)
else:
    print("Attributes of c_feat:", dir(c_feat))

img = Image.new('RGB', (224, 224), color = 'red')
img_in = processor(images=img, return_tensors="pt")
img_feat = model.get_image_features(**img_in)

print("Type of img_feat:", type(img_feat))
if isinstance(img_feat, torch.Tensor):
    print("Shape of img_feat:", img_feat.shape)
else:
    print("Attributes of img_feat:", dir(img_feat))
