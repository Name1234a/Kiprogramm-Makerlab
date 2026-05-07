import torch
from PIL import Image
from torchvision import transforms

ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
model = ckpt["model"]
model.eval()
image = Image.open("Werk.jpg").convert("RGB")
transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
input_tensor = transform(image).unsqueeze(0).half()

with torch.no_grad():
    results = model(input_tensor)

print(type(results))
print(len(results))
for i, item in enumerate(results):
    print("result", i, type(item))
    if hasattr(item, "shape"):
        print(" shape", item.shape)
    if isinstance(item, dict):
        for k, v in item.items():
            print("  key", k, "type", type(v), "shape", getattr(v, "shape", "N/A"))
            if hasattr(v, "shape"):
                print("   first", v.flatten()[:10].tolist())

# Zusätzliche Details
pred = results[0]
print('pred[0].shape', pred[0].shape)
print('pred[0][0, :10]', pred[0, :10].tolist())
print('pred[0][:, :10]', pred[0, :, :10].tolist())

if isinstance(results[1], dict):
    scores = results[1].get('scores')
    boxes = results[1].get('boxes')
    print('scores shape', scores.shape if scores is not None else None)
    print('scores first', scores[0, :, :10].tolist() if scores is not None else None)
    print('boxes shape', boxes.shape if boxes is not None else None)
    print('boxes first', boxes[0, :, :5].tolist() if boxes is not None else None)
