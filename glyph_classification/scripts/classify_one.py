from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image



TEST_IMG = "test_classification/test_11.png"  # CHANGE THIS
CKPT_PATH = "glyph_classification/models/v1/resnet18_v1_stratified_10ep_NOflip.pth"
TOPK = 5



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = Path(CKPT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img_path = Path(TEST_IMG)
    if not img_path.exists():
        raise FileNotFoundError(f"Test image not found: {img_path}")
    
    print("\nImage file:", img_path.name)

    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)  

    topk = min(TOPK, num_classes)
    vals, idxs = torch.topk(probs, k=topk)

    print("Top predictions:")
    for p, i in zip(vals.tolist(), idxs.tolist()):
        print(f"  {class_names[i]}: {p:.3f}")

    pred = idxs[0].item()
    print(f"Predicted class: {class_names[pred]}")


if __name__ == "__main__":
    main()