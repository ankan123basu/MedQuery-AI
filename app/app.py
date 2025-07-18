from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from query_engine import QueryEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'documents')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the query engine
engine = QueryEngine()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('file')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
    
    if not uploaded_files:
        return jsonify({"error": "No valid files uploaded"}), 400
    
    # Process the uploaded files
    try:
        num_chunks = engine.load_documents(uploaded_files)
        return jsonify({
            "message": f"Successfully processed {len(uploaded_files)} file(s) with {num_chunks} chunks",
            "files": [os.path.basename(f) for f in uploaded_files]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        result = engine.query(data['question'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_documents():
    try:
        # Clear the vector store
        if hasattr(engine, 'vector_store') and engine.vector_store is not None:
            try:
                # Try to delete the collection
                collection = engine.vector_store._collection
                if collection is not None:
                    collection.delete(where={"source": {"$ne": ""}})
            except Exception as e:
                print(f"Error clearing vector store: {e}")
            finally:
                engine.vector_store = None
        
        # Clear the documents folder
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Clear the ChromaDB persistent storage
        db_path = os.path.join(os.path.dirname(__file__), 'db')
        if os.path.exists(db_path):
            try:
                import shutil
                shutil.rmtree(db_path)
            except Exception as e:
                print(f"Error removing ChromaDB storage: {e}")
        
        return jsonify({"message": "All documents and vector store cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
