import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API for embeddings
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
except Exception as e:
    print(f"Error configuring Gemini API for embeddings: {e}")

class DocumentProcessor:
    def __init__(self):
        try:
            # Initialize with the latest embedding model
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            logger.info("Document processor initialized with Gemini embeddings")
            
            # Configure Tesseract path - Updated to D: drive location
            tesseract_path = r'D:\\Tesseract-OCR\\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Tesseract path set to: {tesseract_path}")
            else:
                logger.warning(f"Tesseract not found at {tesseract_path}. Please ensure Tesseract is installed correctly.")
                
        except Exception as e:
            logger.error(f"Error initializing document processor: {e}")
            raise
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF, handling both searchable and scanned PDFs.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the document
        """
        logger.info(f"Processing document: {file_path}")
        
        # Check file extension
        if not file_path.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Please upload a PDF or image file.")
        
        # Handle image files directly with OCR
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self._process_image(file_path)
            
        # Handle PDF files
        text = ""
        try:
            # First try to extract text directly
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # If no text was found or very little text, try OCR
            if len(text.strip()) < 100:  # Threshold for considering text as insufficient
                logger.info("Insufficient text found, attempting OCR...")
                ocr_text = self._ocr_pdf(file_path)
                if len(ocr_text) > len(text):
                    text = ocr_text
                    
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            logger.info("Falling back to OCR...")
            text = self._ocr_pdf(file_path)
            
        if not text.strip():
            logger.warning("No text could be extracted from the document")
            return "Error: The document appears to be empty or could not be processed."
            
        logger.info(f"Successfully extracted {len(text)} characters from the document")
        return text
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Dilation and erosion to remove noise
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.dilate(thresh, kernel, iterations=1)
            processed = cv2.erode(processed, kernel, iterations=1)
            
            # Sharpening
            kernel_sharpening = np.array([[-1,-1,-1], 
                                        [-1, 9,-1],
                                        [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel_sharpening)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            # Return original if preprocessing fails
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    def _process_image(self, image_path: str) -> str:
        """
        Process a single image file with enhanced OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        try:
            # Log the start of processing
            logger.info(f"Processing image: {image_path}")
            
            # Open and validate image
            try:
                img = Image.open(image_path)
                if img is None:
                    raise ValueError("Failed to load image")
            except Exception as e:
                logger.error(f"Error opening image {image_path}: {e}")
                return "Error: Could not open the image file."
            
            # Log original image properties
            logger.debug(f"Original image mode: {img.mode}, size: {img.size}")
            
            # Convert to RGB if needed (required for OpenCV)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array for OpenCV processing
            img_np = np.array(img)
            
            # Save original image for debugging
            debug_dir = Path("debug_ocr")
            debug_dir.mkdir(exist_ok=True)
            (debug_dir / "00_original.jpg").write_bytes(cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))[1].tobytes())
            
            # Enhanced preprocessing pipeline
            processed = self._preprocess_image(Image.fromarray(img_np))
            
            # Save preprocessed image for debugging
            cv2.imwrite(str(debug_dir / "01_preprocessed.jpg"), processed)
            
            # Try multiple OCR configurations for better results
            configs = [
                r'--oem 3 --psm 6',  # Assume a single uniform block of text
                r'--oem 1 --psm 3',  # Fully automatic page segmentation, but no OSD
                r'--oem 1 --psm 4',   # Assume a single column of text of variable sizes
                r'--oem 1 --psm 11'   # Sparse text
            ]
            
            best_text = ""
            for i, config in enumerate(configs):
                try:
                    # Try with different preprocessing levels
                    for preprocess_level in ['high', 'medium', 'low']:
                        temp_img = processed.copy()
                        
                        if preprocess_level == 'high':
                            # Additional preprocessing for difficult cases
                            temp_img = cv2.GaussianBlur(temp_img, (3, 3), 0)
                            _, temp_img = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Save intermediate images for debugging
                        cv2.imwrite(str(debug_dir / f"{i+2:02d}_ocr_{preprocess_level}.jpg"), temp_img)
                        
                        # Try OCR with current config
                        page_text = pytesseract.image_to_string(
                            temp_img,
                            config=config,
                            lang='eng'  # Start with English only for better accuracy
                        )
                        
                        # If we got more text than before, keep it
                        if len(page_text.strip()) > len(best_text.strip()):
                            best_text = page_text
                            
                except Exception as e:
                    logger.warning(f"OCR attempt {i+1} failed: {e}")
                    continue
            
            # If we still don't have good text, try with multiple languages
            if len(best_text.strip()) < 20:  # Threshold for considering OCR failed
                logger.info("Trying with multiple languages...")
                try:
                    page_text = pytesseract.image_to_string(
                        processed,
                        config=r'--oem 3 --psm 6',
                        lang='eng+fra+spa+deu'
                    )
                    if len(page_text.strip()) > len(best_text.strip()):
                        best_text = page_text
                except:
                    pass
            
            # Log OCR results
            logger.info(f"Extracted {len(best_text)} characters from image")
            logger.debug(f"Extracted text: {best_text[:200]}..." if best_text else "No text extracted")
            
            if not best_text.strip():
                logger.warning("No text could be extracted from the image")
                return "Error: Could not extract any text from the image. The image may be of poor quality or contain no readable text."
                
            return best_text.strip()
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True)
            return f"Error: Failed to process image - {str(e)}"
    
    def _ocr_pdf(self, file_path: str) -> str:
        """
        Perform OCR on a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the document
        """
        text = ""
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                try:
                    page = doc.load_page(page_num)
                    
                    # Increase resolution for better OCR
                    zoom = 2  # Zoom factor
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Enhance and preprocess image
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.5)
                    processed = self._preprocess_image(img)
                    
                    # Use multiple OCR configurations for better results
                    configs = [
                        r'--oem 3 --psm 6',  # Default
                        r'--oem 1 --psm 3',  # Sparse text
                        r'--oem 1 --psm 4'   # Single column of text
                    ]
                    
                    best_text = ""
                    for config in configs:
                        try:
                            page_text = pytesseract.image_to_string(
                                processed, 
                                config=config,
                                lang='eng+fra+spa+deu'  # Multiple languages
                            )
                            if len(page_text) > len(best_text):
                                best_text = page_text
                        except Exception as e:
                            logger.warning(f"OCR config {config} failed: {e}")
                            continue
                    
                    if best_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{best_text}"
                    
                    logger.info(f"Processed page {page_num + 1}/{total_pages}")
                    
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num + 1}: {page_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            raise
            
        return text.strip()
    
    def process_document(self, file_path: str) -> List[Tuple[str, List[float]]]:
        """
        Process a document and return chunks with embeddings.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of tuples containing (text_chunk, embedding)
        """
        try:
            logger.info(f"Starting to process document: {file_path}")
            
            # Extract text from the document
            text = self.extract_text_from_pdf(file_path)
            
            if not text or "Error:" in text:
                logger.error(f"Failed to extract text from {file_path}")
                return []
            
            # Log first 200 chars for debugging
            logger.debug(f"Extracted text sample: {text[:200]}...")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Document split into {len(chunks)} chunks")
            
            if not chunks:
                logger.error("No text chunks generated from the document")
                return []
            
            # Generate embeddings for each chunk
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embeddings.embed_query(chunk)
                    embeddings.append(embedding)
                    logger.debug(f"Generated embedding for chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {i+1}: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 768)  # Assuming 768D embeddings
            
            logger.info(f"Successfully processed document: {file_path}")
            return list(zip(chunks, embeddings))
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
            return []

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Test with a sample PDF
    # chunks = processor.process_document("path/to/your/document.pdf")
