import os, json, fitz
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_N_SECTIONS = 20
print("[INFO] Loading Sentence-BERT...")
embedder = SentenceTransformer(MODEL_NAME)

# === Section extraction ===
def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        for para in text.split("\n"):
            if 20 < len(para) < 500:
                sections.append({
                    "document": os.path.basename(pdf_path),
                    "page": page_num,
                    "section_title": para.strip()
                })
    return sections

# === Ranking based on persona & job ===
def rank_sections(pdf_paths, persona, job):
    sections = []
    for pdf in pdf_paths:
        sections.extend(extract_sections(pdf))
    query = f"{persona}. Task: {job}"
    print("[INFO] Computing embeddings...")
    query_vec = embedder.encode(query, convert_to_tensor=True)
    section_vecs = embedder.encode([s['section_title'] for s in sections], convert_to_tensor=True)
    scores = util.cos_sim(query_vec, section_vecs)[0]
    for i, s in enumerate(sections):
        s["importance_score"] = float(scores[i])
    ranked = sorted(sections, key=lambda x: x["importance_score"], reverse=True)
    for idx, sec in enumerate(ranked, start=1):
        sec["importance_rank"] = idx
    return ranked[:TOP_N_SECTIONS]

# === Main ===
def process_persona_job(input_dir, output_dir, persona, job):
    pdfs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdf")]
    ranked_sections = rank_sections(pdfs, persona, job)
    result = {
        "metadata": {
            "documents": [os.path.basename(p) for p in pdfs],
            "persona": persona,
            "job_to_be_done": job,
            "timestamp": datetime.now().isoformat()
        },
        "extracted_sections": ranked_sections
    }
    with open(os.path.join(output_dir, "persona_analysis.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Saved persona_analysis.json")

if __name__ == "__main__":
    persona = os.environ.get("PERSONA", "Generic User")
    job = os.environ.get("JOB", "Summarize key insights")
    process_persona_job("/app/input", "/app/output", persona, job)
