<div align="center">

# 🏥 MedQuery AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-44CC11)](https://flask.palletsprojects.com/)
[![AI](https://img.shields.io/badge/AI-Gemini%202.0-FF6F00)](https://ai.google/discover/gemini/)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.335-00A67D)](https://python.langchain.com/)

### Intelligent Medical Document Processing and Query System

</div>

MedQuery AI is an intelligent document processing system designed specifically for medical and insurance documents. It uses advanced AI to extract, process, and answer questions from uploaded documents with high accuracy.
<img width="1917" height="976" alt="Screenshot 2025-07-18 160424" src="https://github.com/user-attachments/assets/51ecf1aa-403e-4e1e-be76-f576ea62f7a6" />

<img width="1919" height="866" alt="Screenshot 2025-07-20 023019" src="https://github.com/user-attachments/assets/6f391d32-c671-48ed-9135-a63dfe46b972" />

<img width="1919" height="888" alt="Screenshot 2025-07-20 024605" src="https://github.com/user-attachments/assets/2986fa3c-bda1-44e8-b678-af1bbd1cd162" />

<img width="1919" height="864" alt="Screenshot 2025-07-20 025012" src="https://github.com/user-attachments/assets/41d0ce74-d213-44ba-8a58-887e75ad6cf9" />

<img width="1775" height="830" alt="image" src="https://github.com/user-attachments/assets/ebd49a09-b59a-4006-a39c-aaca465bce36" />

<img width="1919" height="908" alt="Screenshot 2025-07-18 192344" src="https://github.com/user-attachments/assets/49777167-6183-432b-a7fe-797eaab4b705" />

<img width="1915" height="911" alt="Screenshot 2025-07-18 192739" src="https://github.com/user-attachments/assets/b20a17b2-7f4e-48c8-bee9-ed83fabbe2db" />

<img width="1919" height="907" alt="Screenshot 2025-07-18 183144" src="https://github.com/user-attachments/assets/ab0afa62-536e-40cc-aa0a-a4db14233e99" />

<img width="1786" height="901" alt="image" src="https://github.com/user-attachments/assets/79d30f34-64b1-4c75-921f-cf37eb4cd481" />


## 🚀 Features

- **Document Processing**: Handles PDFs and images (PNG, JPG, JPEG)
- **OCR Capabilities**: Advanced text extraction from scanned documents
- **AI-Powered Q&A**: Accurate responses based on document content
- **Multi-Document Support**: Process and query multiple documents simultaneously
- **Web Interface**: User-friendly web interface for easy interaction

## 🛠 Tech Stack

### Core Technologies

- **Python 3.11+** - Backend language
- **Flask 2.3.3** - Web framework
- **Google Gemini** - AI language model (gemini-2.0-flash-001)
- **ChromaDB** - Vector database for document storage and retrieval
- **Tesseract OCR** - Optical Character Recognition
- **OpenCV** - Image processing

### Key Libraries

- **Google Generative AI** - Direct integration with Gemini models
- **PyMuPDF** - PDF processing
- **Pillow** - Image processing
- **NumPy** - Numerical computations
- **Sentence Transformers** - Text embeddings

### Limited LangChain Usage

- **LangChain Google GenAI** - Lightweight wrapper for Gemini model
- **LangChain Core** - Basic prompt templates and output parsing
- **LangChain Community** - ChromaDB vector store integration
- **Text Splitting** - Document chunking functionality

## 📊 Performance Metrics

### Document Processing

- **OCR Accuracy**:  
  - Searchable PDFs: ~99% (native text extraction)  
  - Clean scanned documents: ~85-90%  
  - Complex/layout documents: ~70-80%  
  - Handwritten text: ~50-60% (varies significantly with handwriting quality)
- **Processing Speed**:
  - Text-based PDFs: ~1-3 seconds per page
  - Scanned documents: ~3-8 seconds per page
  - Images: ~2-6 seconds per image
- **Supported Languages**: English (primary), with basic support for other Latin-based languages (accuracy may vary)

### AI Model

- **Model**: Gemini 2.0 Flash (gemini-2.0-flash-001)
- **Context Window**: 128K tokens
- **Response Time**:  
  - Simple queries: 1-3s  
  - Complex queries: 3-8s  
  - Document analysis: 5-15s
- **Temperature**: 0.3 (focused on accuracy)
- **Limitations**:
  - May struggle with very specific medical terminology
  - Accuracy depends on source document quality
  - Best results with well-structured documents

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 500MB free space (plus document storage)
- **Tesseract OCR**: Required for image processing

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Tesseract OCR installed and in system PATH
- Google API key for Gemini

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements_updated.txt
   ```
3. Set up your `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Application
```bash
python app/app.py
```

## 📈 Performance Benchmarks

### Document Processing

| Document Type | Processing Time (avg) | Accuracy | Notes |
|---------------|----------------------|----------|-------|
| Searchable PDF | 1-3s/page | 99% | Native text extraction with PyMuPDF |
| Scanned PDF (300dpi) | 4-8s/page | 85-90% | Depends on scan quality |
| Image (JPG/PNG) | 2-6s/image | 80-90% | High variance based on quality |
| Handwritten Notes | 10-15s/page | 50-60% | Highly variable, best with clear print |
| Forms/Tables | 5-10s/page | 70-80% | Structure may not be preserved |

### Query Performance

| Query Type | Response Time | Accuracy | Notes |
|------------|--------------|----------|-------|
| Fact Retrieval | 1-3s | 90-95% | Best for direct information |
| Summary | 2-4s | 85-90% | Quality depends on document structure |
| Complex Analysis | 3-8s | 75-85% | May require multiple document passes |
| Multi-document | 5-15s | 70-85% | Depends on document count and size |

## 🔍 Architecture

### Document Processing Pipeline

1. **Document Ingestion**:
   - Upload documents through the web interface
   - Documents are processed using PyMuPDF (PDFs) or Tesseract OCR (images)
   - Text is split into manageable chunks using LangChain's text splitter

2. **Vector Storage**:
   - Text chunks are converted to vector embeddings using Google's Gemini embeddings
   - Vectors are stored in ChromaDB with basic metadata
   - Direct ChromaDB integration (not through full LangChain stack)

3. **Query Flow**:
   - User questions are processed through a lightweight LangChain wrapper
   - Similarity search is performed directly against ChromaDB
   - Gemini generates responses using minimal LangChain abstractions
   - Raw Gemini responses are parsed and formatted for display

### Key Implementation Notes

- Uses direct Gemini API calls where possible
- LangChain components are used only where they provide clear value
- Custom document processing pipeline for better control over performance
- Minimal abstraction layer between Gemini and the application

## 🛡️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Ankan Basu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ✨ Tech Stack

<div align="center">

| Category | Technologies |
|----------|--------------|
| **Backend** | Python, Flask |
| **AI/ML** | Google Gemini, LangChain |
| **OCR** | Tesseract, OpenCV |
| **Vector DB** | ChromaDB |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | (Specify if any) |

</div>

## 📬 Contact

For support, questions, or collaboration opportunities, please feel free to reach out:

- 📧 Email: [ankanbasu10@gmail.com]
- 💼 LinkedIn: [https://www.linkedin.com/in/ankan-basu-595152271/]
- 🌐 GitHub: [@ankan123basu](https://github.com/ankan123basu)

## 👨‍💻 Developer

<div align="center">

Made with ❤️ by **Ankan Basu**  
Undergraduate Student  
[Lovely Professional University](https://www.lpu.in/)  
Punjab, India

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ankan123basu)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ankan-basu-595152271/)
</div>
