# app.py — cleaned, deduplicated and fixed
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    PreTrainedModel
)
import fitz  # PyMuPDF
from docx import Document

# ---------------- Flask setup ----------------
app = Flask(__name__)
app.secret_key = "your-secret-key-here"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- device & global models ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Essay embedding model
essay_model = SentenceTransformer("all-MiniLM-L6-v2")

# CodeBERT for semantic code embeddings (used if available)
codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
codebert_model.eval()

# Desklib AI detector model wrapper (PreTrainedModel subclass)
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # instantiate base transformer from config
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs[0]  # (batch, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_emb / sum_mask
        logits = self.classifier(pooled)
        return {"logits": logits}

# Load Desklib detector (model directory on HF)
AI_MODEL_DIR = "desklib/ai-text-detector-v1.01"
ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_DIR)
ai_model = DesklibAIDetectionModel.from_pretrained(AI_MODEL_DIR).to(device)
ai_model.eval()

# ---------------- file text extraction ----------------
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with fitz.open(file_path) as pdf:
                for p in pdf:
                    text += p.get_text("text")
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            # txt, md, code files, etc.
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
    return text.strip()

# ---------------- AI detection ----------------
def detect_ai_likelihood(text, max_len=1024, threshold_model_max=768):
    """
    Returns probability in [0.0, 100.0] that `text` is AI-generated using desklib model.
    """
    if not text or not text.strip():
        return 0.0
    try:
        # truncate sensibly: the desklib model supports long max_length, but we keep it reasonable
        encoded = ai_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        for k in encoded:
            encoded[k] = encoded[k].to(device)

        with torch.no_grad():
            outputs = ai_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            logits = outputs["logits"]  # shape (batch, 1)
            prob = torch.sigmoid(logits).squeeze().item()  # in [0,1]
        return round(prob * 100.0, 2)
    except Exception as e:
        print(f"[AI Detector Error] {e}")
        return 0.0

# ---------------- essay similarity ----------------
def essay_similarity(path1, path2):
    t1 = extract_text_from_file(path1)
    t2 = extract_text_from_file(path2)
    if not t1 or not t2:
        return 0.0
    try:
        embeddings = essay_model.encode([t1, t2], convert_to_tensor=True)
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(sim * 100.0, 2)
    except Exception as e:
        print(f"[Essay similarity error] {e}")
        return 0.0

# ---------------- code normalization & similarity primitives ----------------
def normalize_multilang_code(code, ext=None):
    # strip common comment patterns and normalize strings/numbers
    code = re.sub(r'/\*[\s\S]*?\*/', " ", code)   # block comments
    code = re.sub(r'//.*', " ", code)             # c/c++/js single-line
    code = re.sub(r'#.*', " ", code)              # python/shell
    code = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\')', " <STR> ", code, flags=re.DOTALL)
    code = re.sub(r'(\".*?\"|\'.*?\')', " <STR> ", code)
    code = re.sub(r'\b\d+(\.\d+)?\b', " <NUM> ", code)
    code = re.sub(r'\s+', " ", code)
    return code.strip()

def lexical_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def tfidf_char_similarity(a, b):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))
    tf = v.fit_transform([a, b])
    return float(cosine_similarity(tf[0:1], tf[1:2])[0][0])

def jaccard_token_similarity(a, b):
    t1 = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', a))
    t2 = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', b))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def codebert_similarity(a, b):
    try:
        inputs = codebert_tokenizer([a, b], return_tensors="pt", truncation=True, padding=True, max_length=512)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        with torch.no_grad():
            out = codebert_model(**inputs).last_hidden_state.mean(dim=1)  # (2, hidden)
        sim = F.cosine_similarity(out[0].unsqueeze(0), out[1].unsqueeze(0)).item()
        return float(sim)
    except Exception as e:
        # fail gracefully; return 0 semantic similarity
        print(f"[CodeBERT error] {e}")
        return 0.0

def code_similarity(path1, path2, use_codebert=True):
    try:
        ext1 = os.path.splitext(path1)[1].lower()
        ext2 = os.path.splitext(path2)[1].lower()

        with open(path1, "r", encoding="utf-8", errors="ignore") as f:
            c1 = f.read()
        with open(path2, "r", encoding="utf-8", errors="ignore") as f:
            c2 = f.read()

        c1n = normalize_multilang_code(c1, ext1)
        c2n = normalize_multilang_code(c2, ext2)

        lex = lexical_similarity(c1n, c2n)
        tfidf = tfidf_char_similarity(c1n, c2n)
        jacc = jaccard_token_similarity(c1n, c2n)
        sem = codebert_similarity(c1n, c2n) if use_codebert else 0.0

        # Weighted blend (tunable)
        score = (0.2 * lex + 0.35 * tfidf + 0.35 * jacc + 0.1 * sem) * 100.0
        return round(score, 2)
    except Exception as e:
        print(f"[code_similarity error] {e}")
        return 0.0

# ---------------- helper debug function ----------------
def debug_compare_files(f1, f2):
    p1 = os.path.join(UPLOAD_FOLDER, f1)
    p2 = os.path.join(UPLOAD_FOLDER, f2)
    t1 = extract_text_from_file(p1)
    t2 = extract_text_from_file(p2)
    print("---- preview", f1)
    print(t1[:800])
    print("---- preview", f2)
    print(t2[:800])
    print("lex:", lexical_similarity(t1, t2))
    try:
        print("tfidf:", tfidf_char_similarity(t1, t2))
    except Exception as e:
        print("tfidf error", e)

# ---------------- plagiarism detection orchestration ----------------
def detect_plagiarism(files):
    results = {}
    file_data = {}
    pair_sims = {}

    # read all files once and compute ai-likelihood
    for fn in files:
        path = os.path.join(UPLOAD_FOLDER, fn)
        ext = os.path.splitext(path)[1].lower()
        content = extract_text_from_file(path)
        file_data[fn] = {"path": path, "ext": ext, "content": content}
        ai_score = detect_ai_likelihood(content)
        results[fn] = {
            "ai_likelihood": ai_score,
            "ai_risk_level": "low" if ai_score < 30 else "medium" if ai_score < 60 else "high",
            "similarities": []
        }

    # pairwise similarities (compute only once per pair)
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            f1, f2 = files[i], files[j]
            d1, d2 = file_data[f1], file_data[f2]
            sim = 0.0
            if not d1["content"] or not d2["content"]:
                sim = 0.0
            elif any(e in [".txt", ".md", ".docx", ".pdf"] for e in (d1["ext"], d2["ext"])):
                # treat at least one as essay
                sim = essay_similarity(d1["path"], d2["path"])
            else:
                sim = code_similarity(d1["path"], d2["path"])
            pair_sims[(f1, f2)] = pair_sims[(f2, f1)] = sim

    # attach similarities back to results
    for f1 in files:
        for f2 in files:
            if f1 == f2:
                continue
            sim = pair_sims.get((f1, f2), 0.0)
            status = "high" if sim > 80 else "medium" if sim > 50 else "low"
            results[f1]["similarities"].append({
                "compared_with": f2,
                "similarity": sim,
                "status": status
            })
    return results

# ---------------- routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    if "files[]" not in request.files:
        return redirect(request.url)
    uploaded = request.files.getlist("files[]")
    names = []
    for f in uploaded:
        if f.filename:
            target = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(target)
            names.append(f.filename)
    results = detect_plagiarism(names)
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=False)
