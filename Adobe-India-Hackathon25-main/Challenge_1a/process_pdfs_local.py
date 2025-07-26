import os, json, re, fitz
import torch
import torchvision.transforms as T
from PIL import Image

MODEL_PATH = "mobilenetv2_heading.pth"
device = "cpu"
cnn_threshold = 0.5

# Load trained CNN
heading_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
heading_model.eval()

transform = T.Compose([
    T.Resize((64, 64)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

def cnn_classify_block(page, bbox):
    pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=72)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = heading_model(img_tensor)
    return torch.sigmoid(out).item()

def classify_heading(text, size, font, position, page_height, cnn_score):
    score = 0
    if size > 20: score += 2
    elif size > 16: score += 1
    if text.isupper() or "Bold" in font: score += 1
    if position < page_height * 0.3: score += 1
    if re.match(r'^(\d+(\.\d+)*\s+|Chapter|Section)', text): score += 1
    if cnn_score > cnn_threshold: score += 2
    if score >= 5: return "H1"
    elif score >= 3: return "H2"
    elif score >= 2: return "H3"
    return None

def extract_outline(pdf_path, mode="strict"):
    doc = fitz.open(pdf_path)
    title, outline = "", []
    # Extract title (largest text on first page)
    first_page = doc[0]
    candidates = [(span['size'], span['text'])
                  for b in first_page.get_text("dict")["blocks"]
                  for l in b.get("lines", []) for span in l.get("spans", [])]
    if candidates:
        title = max(candidates, key=lambda x: x[0])[1].strip()

    # Extract all potential headings
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            y_pos = b.get("bbox", [0, 0, 0, 0])[1]
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    text = s['text'].strip()
                    if not text or len(text) > 150:
                        continue
                    cnn_score = cnn_classify_block(page, b.get("bbox"))
                    level = classify_heading(
                        text=text,
                        size=s['size'],
                        font=s['font'],
                        position=y_pos,
                        page_height=page.rect.height,
                        cnn_score=cnn_score
                    )
                    if level:
                        outline.append({"level": level, "text": text, "page": page_num})

    # === Post-filter for Adobe compliance ===
    if mode == "strict":
        # Remove fragments: numeric only or very short (<3 words)
        filtered = [h for h in outline if len(h['text'].split()) >= 3 and not re.match(r'^\d+\.?$', h['text'])]
        # If mostly fragments or very few headings → treat as form
        if len(filtered) < 3:
            outline = []
        else:
            outline = filtered

    return {"title": title, "outline": outline}

    doc = fitz.open(pdf_path)
    title, outline = "", []
    # Extract title
    first_page = doc[0]
    candidates = [(span['size'], span['text'])
                  for b in first_page.get_text("dict")["blocks"]
                  for l in b.get("lines", []) for span in l.get("spans", [])]
    if candidates:
        title = max(candidates, key=lambda x: x[0])[1].strip()

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            y_pos = b.get("bbox", [0, 0, 0, 0])[1]
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    text = s['text'].strip()
                    if not text or len(text) > 150: 
                        continue
                    cnn_score = cnn_classify_block(page, b.get("bbox"))
                    level = classify_heading(
                        text=text,
                        size=s['size'],
                        font=s['font'],
                        position=y_pos,
                        page_height=page.rect.height,
                        cnn_score=cnn_score
                    )
                    if level:
                        outline.append({"level": level, "text": text, "page": page_num})
    return {"title": title, "outline": outline}

def process_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith(".pdf"):
            print(f"[INFO] Processing {file}...")
            pdf_path = os.path.join(input_dir, file)
            result = extract_outline(pdf_path)
            out_path = os.path.join(output_dir, file.replace(".pdf", ".json"))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[DONE] Saved {out_path}")

if __name__ == "__main__":
    input_dir = "./input"
    output_dir = "./output"
    print(f"[INFO] Reading PDFs from: {input_dir}")
    print(f"[INFO] Writing JSON outputs to: {output_dir}")
    process_pdfs(input_dir, output_dir)
