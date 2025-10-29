import os
import re
import ast
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import fitz  # PyMuPDF for PDF text extraction
from docx import Document  # for DOCX text extraction (optional)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load models once
essay_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load UniXcoder (for main code similarity)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel
# GraphCodeBERT model for semantic code embeddings
graphcodebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
graphcodebert_model = RobertaModel.from_pretrained("microsoft/graphcodebert-base").to(device)


# ---------------- File Text Extraction ----------------
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text("text")

        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif ext in [".txt", ".md", ".py", ".js", ".java", ".cpp", ".c", ".php"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
        text = ""

    return text.strip()

# ---------------- AI Detection Heuristics ----------------
def detect_ai_likelihood(text):
    score, total_checks = 0, 0
    lines = text.split('\n')

    # Check 1: Excessive comments
    total_checks += 1
    comment_lines = sum(1 for l in lines if l.strip().startswith(('#', '//')))
    if len(lines) > 0 and comment_lines / len(lines) > 0.3:
        score += 1

    # Check 2: Over-descriptive or consistent naming
    total_checks += 1
    var_names = re.findall(r'\b[a-z_][a-z0-9_]*\b', text.lower())
    if var_names:
        long_names = [v for v in var_names if len(v) > 15]
        if len(long_names) / len(var_names) > 0.2:
            score += 1

    # Check 3: Too-perfect indentation
    total_checks += 1
    indented = [l for l in lines if l.startswith('    ')]
    if len(indented) > 5 and all(re.match(r'^(    )+\S', l) for l in indented):
        score += 0.5

    return round((score / total_checks) * 100, 2)

# ---------------- Essay Similarity ----------------
def essay_similarity(file1, file2):
    text1 = extract_text_from_file(file1)
    text2 = extract_text_from_file(file2)
    if not text1 or not text2:
        return 0.0

    embeddings = essay_model.encode([text1, text2], batch_size=2, convert_to_tensor=True)
    sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(sim * 100, 2)

# ---------------- Code Similarity (graphcodebert) ----------------
#import torch
import torch.nn.functional as F
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Improved GraphCodeBERT Similarity ----------------
# ---------------- Improved GraphCodeBERT + AST Hybrid Similarity ----------------
import ast
import torch.nn.functional as F
import re
def normalize_ast(code):
    """Convert Python code to an AST structure string (logic-level fingerprint)."""
    try:
        # Normalize: remove docstrings
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    node.body = node.body[1:]
        
        # Convert to string, ignoring specific field values
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:
        return ""  # fallback for syntax errors or non-Python code

def graphcodebert_similarity(code1, code2):
    def get_embedding(text):
        # Clean comments & whitespace (this part is good)
        text = re.sub(r'(#|//).*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
             # Handle empty code after cleaning
            return torch.zeros((1, 768), device=device)

        inputs = graphcodebert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = graphcodebert_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    # --- Semantic Embeddings (What the code *does*) ---
    emb1 = get_embedding(code1)
    emb2 = get_embedding(code2)
    # Get similarity as a float between 0.0 and 1.0
    sim_semantic = F.cosine_similarity(emb1, emb2).item()

    # --- Structural Fingerprint via AST (How the code is *built*) ---
    ast1_str = normalize_ast(code1)
    ast2_str = normalize_ast(code2)
    
    sim_ast = 0.0
    # Check if both files successfully parsed into an AST
    ast_check_possible = bool(ast1_str and ast2_str)
    
    if ast_check_possible:
        emb_ast1 = get_embedding(ast1_str)
        emb_ast2 = get_embedding(ast2_str)
        # Get AST similarity as a float between 0.0 and 1.0
        sim_ast = F.cosine_similarity(emb_ast1, emb_ast2).item()
    
    # --- New Hybrid Score Logic ---
    final_sim = 0.0
    if ast_check_possible:
        # **THIS IS THE FIX**
        # We MULTIPLY the scores.
        # If structure is different (low sim_ast), the final score
        # is dragged down, even if semantics are similar.
        final_sim = sim_semantic * sim_ast
    else:
        # Fallback for syntax errors or non-Python code.
        # We can only trust the semantic score.
        final_sim = sim_semantic

    return round(final_sim * 100, 2)

# ---------------- Plagiarism Detection ----------------
def detect_plagiarism(files_to_check):
    results = {}
    file_data = {} # Will store content and extension

    # 1. Read all files and their content ONCE
    for file_name in files_to_check:
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        ext = os.path.splitext(file_path)[1].lower()
        content = extract_text_from_file(file_path)
        file_data[file_name] = {
            'content': content,
            'ext': ext
        }

    # 2. Process and compare
    for i, file_name in enumerate(files_to_check):
        data = file_data[file_name]
        content = data['content']
        ext = data['ext']
        
        ai_likelihood = detect_ai_likelihood(content)
        results[file_name] = {
            'ai_likelihood': ai_likelihood,
            'ai_risk_level': 'low' if ai_likelihood < 30 else 'medium' if ai_likelihood < 60 else 'high',
            'similarities': []
        }

        # Compare against all other files
        for j, other_file in enumerate(files_to_check):
            if i == j:
                continue
            
            other_data = file_data[other_file]
            other_content = other_data['content']
            other_ext = other_data['ext'] # Use this for the check

            sim = 0.0
            if not content or not other_content:
                sim = 0.0
            elif ext in ['.txt', '.md', '.docx', '.pdf']:
                # Use the new text-based essay function
                sim = essay_similarity(content, other_content)
            else:
                # *** THE FIX: Call the hybrid GraphCodeBERT + AST function ***
                sim = graphcodebert_similarity(content, other_content) 

            status = 'high' if sim > 80 else 'medium' if sim > 50 else 'low'
            results[file_name]['similarities'].append({
                'compared_with': other_file,
                'similarity': sim,
                'status': status
            })

    return results

# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return redirect(request.url)

    uploaded_files = request.files.getlist('files[]')
    current_upload_files = []
    for file in uploaded_files:
        if file.filename:
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            current_upload_files.append(file.filename)

    results = detect_plagiarism(current_upload_files)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=False)
