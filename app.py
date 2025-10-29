import os
import re
import ast
import csv
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from sentence_transformers import CrossEncoder
import torch

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- AI Detection Patterns ----------------
def detect_ai_likelihood(code):
    """
    Analyzes code for AI-written characteristics
    Returns a percentage likelihood (0-100)
    """
    score = 0
    total_checks = 0
    
    # Check 1: Excessive comments (AI often over-comments)
    total_checks += 1
    lines = code.split('\n')
    comment_lines = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))
    if len(lines) > 0:
        comment_ratio = comment_lines / len(lines)
        if comment_ratio > 0.3:  # More than 30% comments
            score += 1
    
    # Check 2: Perfect naming conventions (unusually consistent)
    total_checks += 1
    var_names = re.findall(r'\b[a-z_][a-z0-9_]*\b', code.lower())
    if var_names:
        snake_case = sum(1 for v in var_names if '_' in v and v.islower())
        if len(var_names) > 5 and snake_case / len(var_names) > 0.8:
            score += 0.5
    
    # Check 3: Overly descriptive variable names
    total_checks += 1
    long_names = [v for v in var_names if len(v) > 15]
    if len(var_names) > 0 and len(long_names) / len(var_names) > 0.2:
        score += 1
    
    # Check 4: Type hints in Python (common in AI-generated code)
    total_checks += 1
    if '->' in code or ': int' in code or ': str' in code or ': float' in code:
        score += 0.5
    
    # Check 5: Docstrings presence (AI loves docstrings)
    total_checks += 1
    docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
    if docstring_count > 2:
        score += 1
    
    # Check 6: Try-except blocks (AI tends to add defensive coding)
    total_checks += 1
    try_count = code.count('try:')
    if try_count > len(lines) / 50:  # More than 1 try per 50 lines
        score += 0.5
    
    # Check 7: Consistent indentation (perfect 4 spaces)
    total_checks += 1
    indented_lines = [line for line in lines if line.startswith('    ')]
    if len(indented_lines) > 5:
        irregular_indent = sum(1 for line in indented_lines if not re.match(r'^(    )+\S', line))
        if irregular_indent == 0:
            score += 0.5
    
    # Check 8: Generic function names like 'process_data', 'calculate_result'
    total_checks += 1
    generic_names = ['process', 'calculate', 'handle', 'manage', 'execute', 'perform']
    func_names = re.findall(r'def\s+([a-z_][a-z0-9_]*)', code)
    if func_names:
        generic_count = sum(1 for name in func_names if any(g in name for g in generic_names))
        if generic_count / len(func_names) > 0.3:
            score += 1
    
    # Convert to percentage
    ai_likelihood = (score / total_checks) * 100
    return round(ai_likelihood, 2)

# ---------------- AST-based Structural Extraction ----------------
def extract_ast_structure(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    structure_tokens = []

    for node in ast.walk(tree):
        # Node type
        structure_tokens.append(type(node).__name__)

        # Function definitions
        if isinstance(node, ast.FunctionDef):
            structure_tokens.append(f"FUNC_{node.name}")

        # Variable names
        elif isinstance(node, ast.Name):
            structure_tokens.append(f"VAR_{node.id}")

        # Attribute access (like obj.method)
        elif isinstance(node, ast.Attribute):
            structure_tokens.append(f"ATTR_{node.attr}")

    return ' '.join(structure_tokens)


# ---------------- Fallback Code Normalization ----------------
def fallback_code_similarity(file1, file2):     
    def normalize_code(code):
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'\b[_a-zA-Z]\w*\b', lambda m: m.group(0) if m.group(0) in [
    'if', 'for', 'while', 'return', 'def', 'class', 'import', 'from'
] else 'VAR', code)

        return code

    with open(file1, 'r', encoding='utf-8', errors='ignore') as f1, \
         open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
        code1 = normalize_code(f1.read())
        code2 = normalize_code(f2.read())

    if not code1.strip() or not code2.strip():
        return 0.0

    vectorizer = TfidfVectorizer(min_df=1)
    try:
        vectors = vectorizer.fit_transform([code1, code2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        return 0.0

# ---------------- Code Similarity (AST + Fallback) ----------------
def code_similarity(file1, file2):
    ext = os.path.splitext(file1)[1].lower()

    # Python files → AST similarity
    if ext == '.py':
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1, \
                 open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                code1 = f1.read()
                code2 = f2.read()

            structure1 = extract_ast_structure(code1)
            structure2 = extract_ast_structure(code2)

            if not structure1 or not structure2:
                return fallback_code_similarity(file1, file2)

            vectorizer = TfidfVectorizer(min_df=1)
            vectors = vectorizer.fit_transform([structure1, structure2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return round(similarity * 100, 2)
        except Exception as e:
            print(f"Error in AST similarity: {e}")
            return fallback_code_similarity(file1, file2)

    # Non-Python code → fallback
    else:
        return fallback_code_similarity(file1, file2)

# ---------------- Plagiarism Detection ----------------
def detect_plagiarism(files_to_check):
    """Only check the files that were just uploaded"""
    results = {}
    
    # Calculate similarities between uploaded files
    for i in range(len(files_to_check)):
        file_name = files_to_check[i]
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        
        # Read file content for AI detection
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
        except:
            code_content = ""
        
        ai_likelihood = detect_ai_likelihood(code_content)
        
        # Initialize file report
        results[file_name] = {
            'ai_likelihood': ai_likelihood,
            'similarities': [],
            'ai_risk_level': 'low' if ai_likelihood < 30 else ('medium' if ai_likelihood < 60 else 'high'),
            'ai_color': '#28a745' if ai_likelihood < 30 else ('#ffc107' if ai_likelihood < 60 else '#dc3545')
        }
        
        # Compare with other uploaded files
        for j in range(len(files_to_check)):
            if i != j:
                file1 = os.path.join(UPLOAD_FOLDER, files_to_check[i])
                file2 = os.path.join(UPLOAD_FOLDER, files_to_check[j])
                ext1 = os.path.splitext(file1)[1].lower()
                if ext1 in ['.txt', '.docx', '.md']:
                    sim = essay_similarity(file1, file2)
                else:
                    sim = code_similarity(file1, file2)

                
                # Determine status
                if sim > 80:
                    status = 'high'
                elif sim > 50:
                    status = 'medium'
                else:
                    status = 'low'
                
                results[file_name]['similarities'].append({
                    'compared_with': files_to_check[j],
                    'similarity': sim,
                    'status': status
                })

    return results
# ---------------- Essay Similarity (AI-based) ----------------
def essay_similarity(file1, file2):
    model = CrossEncoder('cross-encoder/stsb-roberta-large')
    with open(file1, 'r', encoding='utf-8', errors='ignore') as f1, \
         open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
        text1 = f1.read()
        text2 = f2.read()

    # Limit text length if huge
    max_chars = 2000
    text1 = text1[:max_chars]
    text2 = text2[:max_chars]

    # Predict similarity score
    score = model.predict([(text1, text2)])[0]
    return round(score * 100, 2)

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
    
    # Save uploaded files and track their names
    for file in uploaded_files:
        if file.filename:
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            current_upload_files.append(file.filename)
    
    # Only analyze the files from this upload session
    results = detect_plagiarism(current_upload_files)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)