import requests
import PyPDF2
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Tuple
import re
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'document_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self, target_words: List[str], similarity_threshold: float = 0.8):
        logger.info(f"Initializing DocumentAnalyzer with target words: {target_words}")
        logger.info(f"Similarity threshold set to: {similarity_threshold}")
        
        # Load the model and tokenizer
        logger.info("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Model and tokenizer loaded successfully")
        
        self.target_words = target_words
        self.similarity_threshold = similarity_threshold
        
    def download_and_parse_pdf(self, url: str) -> str:
        """Download PDF from URL and extract text"""
        logger.info(f"Downloading PDF from URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
            logger.info("PDF downloaded successfully")
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            text = ""
            for i, page in enumerate(pdf_reader.pages, 1):
                logger.info(f"Processing page {i}/{total_pages}")
                text += page.extract_text()
            
            logger.info(f"PDF parsing complete. Extracted {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Error downloading or parsing PDF: {str(e)}")
            raise
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """Get embedding for a single word"""
        logger.debug(f"Getting embedding for word: {word}")
        try:
            inputs = self.tokenizer(word, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return torch.mean(outputs.last_hidden_state, dim=1).squeeze()
        except Exception as e:
            logger.error(f"Error getting embedding for word '{word}': {str(e)}")
            raise
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words"""
        logger.debug(f"Calculating similarity between '{word1}' and '{word2}'")
        try:
            emb1 = self.get_word_embedding(word1)
            emb2 = self.get_word_embedding(word2)
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            logger.debug(f"Similarity score: {similarity:.4f}")
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise
    
    def get_context(self, text: str, position: int, window: int = 100) -> str:
        """Extract context around a position with specified window size"""
        logger.debug(f"Extracting context around position {position} with window size {window}")
        words = text.split()
        start = max(0, position - window)
        end = min(len(words), position + window + 1)
        context = ' '.join(words[start:end])
        logger.debug(f"Extracted context of {len(context)} characters")
        return context
    
    def analyze_document(self, url: str) -> List[str]:
        """Main function to analyze document and find important sections"""
        start_time = time.time()
        logger.info("Starting document analysis")
        
        # Download and parse PDF
        text = self.download_and_parse_pdf(url)
        words = text.split()
        total_words = len(words)
        logger.info(f"Document contains {total_words} words")
        
        important_sections = []
        processed_words = 0
        last_progress = 0
        
        # Process each word in the document
        for i, doc_word in enumerate(words):
            # Log progress every 5%
            progress = (i / total_words) * 100
            if int(progress) > last_progress and int(progress) % 5 == 0:
                logger.info(f"Processing progress: {int(progress)}% ({i}/{total_words} words)")
                last_progress = int(progress)
            
            doc_word = re.sub(r'[^\w\s]', '', doc_word.lower())
            if not doc_word:
                continue
            
            processed_words += 1
            
            # Compare with target words
            for target_word in self.target_words:
                similarity = self.calculate_similarity(doc_word, target_word)
                if similarity >= self.similarity_threshold:
                    logger.info(f"Found high similarity match: '{doc_word}' -> '{target_word}' (score: {similarity:.4f})")
                    context = self.get_context(text, i)
                    important_sections.append({
                        'matched_word': doc_word,
                        'target_word': target_word,
                        'similarity': similarity,
                        'context': context
                    })
                    break  # Move to next word once we find a match
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Document analysis complete in {processing_time:.2f} seconds")
        logger.info(f"Processed {processed_words} words and found {len(important_sections)} important sections")
        
        return important_sections

def main():
    # Example usage
    target_words = ['bcorp', 'governance', 'compliance']  # Replace with your target words
    pdf_url = "https://3l7ggmaw4w.ufs.sh/f/PXsP0owYxqWm1lfSsOHURCBXruTgAdvPmO6F4oSqncLyIEZj"  # Replace with your PDF URL
    
    logger.info("Starting main execution")
    logger.info(f"Target words: {target_words}")
    logger.info(f"PDF URL: {pdf_url}")
    
    try:
        analyzer = DocumentAnalyzer(target_words)
        results = analyzer.analyze_document(pdf_url)
        
        # Print results
        logger.info(f"Found {len(results)} matching sections")
        for i, result in enumerate(results, 1):
            logger.info(f"\nImportant Section {i}:")
            logger.info(f"Matched word: {result['matched_word']}")
            logger.info(f"Target word: {result['target_word']}")
            logger.info(f"Similarity: {result['similarity']:.2f}")
            logger.info(f"Context:\n{result['context']}\n")
            logger.info("-" * 80)
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 