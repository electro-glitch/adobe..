import os, json, fitz, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

PDF_DIR = "sample_dataset/pdfs"
LABEL_DIR = "sample_dataset/outputs"
OUT_PATH = "mobilenetv2_heading.pth"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class HeadingDataset(Dataset):
    def __init__(self, pdf_dir, label_dir, transform):
        self.samples = []
        self.transform = transform
        for pdf in os.listdir(pdf_dir):
            if not pdf.endswith(".pdf"): continue
            base = pdf.replace(".pdf", ".json")
            with open(os.path.join(label_dir, base), 'r', encoding='utf-8') as f:
                labels = json.load(f)
            heading_texts = set([item["text"] for item in labels["outline"]])
            doc = fitz.open(os.path.join(pdf_dir, pdf))
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            text = s['text'].strip()
                            if not text: continue
                            label = 1 if text in heading_texts else 0
                            self.samples.append((page, b["bbox"], label))
        print(f"[INFO] Dataset size: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        page, bbox, label = self.samples[idx]
        pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=72)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return self.transform(img), torch.tensor(label, dtype=torch.float32)

dataset = HeadingDataset(PDF_DIR, LABEL_DIR, transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# MobileNetV2
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Sequential(nn.Linear(model.last_channel, 1), nn.Sigmoid())

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = "cpu"
model.to(device)

for epoch in range(5):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model, OUT_PATH)
print(f"[DONE] Model saved to {OUT_PATH}")
