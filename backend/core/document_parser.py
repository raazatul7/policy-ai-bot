"""
Document Parser Module for AI Policy Query System

This module handles parsing of insurance policy documents (PDF and DOCX formats)
and splits them into manageable text chunks for embedding and retrieval.
"""

import os
import re
from typing import List, Optional
from pathlib import Path

import pdfplumber
from docx import Document


class DocumentParser:
    """
    Handles parsing of PDF and DOCX documents into text chunks.
    
    This class provides methods to extract text from policy documents
    and split them into smaller chunks suitable for embedding and retrieval.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the document parser.
        
        Args:
            chunk_size: Target number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def parse_document(self, file_path: str) -> List[str]:
        """
        Parse a document and return text chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
            RuntimeError: If file is empty or corrupted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size (prevent memory issues)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise RuntimeError("File is empty")
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise RuntimeError("File too large (max 100MB)")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            text = self._parse_pdf(file_path)
        elif file_extension == '.docx':
            text = self._parse_docx(file_path)
        elif file_extension == '.txt':
            text = self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Validate extracted text
        if not text or not text.strip():
            raise RuntimeError("No text content could be extracted from the document")
        
        # Clean and chunk the extracted text
        cleaned_text = self._clean_text(text)
        chunks = self._create_chunks(cleaned_text)
        
        if not chunks:
            raise RuntimeError("No valid text chunks could be created from the document")
        
        return chunks
    
    def _parse_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        text_content = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from the page
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Add page marker for reference tracking
                        text_content.append(f"\n--- Page {page_num} ---\n")
                        text_content.append(page_text)
                        
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF file: {str(e)}")
        
        return "\n".join(text_content)
    
    def _parse_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text as a single string
        """
        text_content = []
        
        try:
            doc = Document(file_path)
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Skip empty paragraphs
                    text_content.append(paragraph.text.strip())
                    
        except Exception as e:
            raise RuntimeError(f"Error parsing DOCX file: {str(e)}")
        
        return "\n".join(text_content)
    
    def _parse_txt(self, file_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            File content as a string
        """
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                if content.strip():  # Check if we got meaningful content
                    return content
                    
            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                raise RuntimeError(f"Error parsing TXT file: {str(e)}")
        
        # If all encodings fail, try with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return content
        except Exception as e:
            raise RuntimeError(f"Error parsing TXT file: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
        text = text.strip()
        
        return text
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately the specified token size.
        
        Args:
            text: Clean text to be chunked
            
        Returns:
            List of text chunks
        """
        # Simple word-based tokenization (approximation)
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            # Calculate chunk end position
            chunk_end = min(i + self.chunk_size, len(words))
            
            # Extract chunk words
            chunk_words = words[i:chunk_end]
            chunk_text = ' '.join(chunk_words)
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Move to next chunk with overlap
            i += self.chunk_size - self.overlap
        
        return chunks
    
    def get_document_metadata(self, file_path: str) -> dict:
        """
        Extract metadata from the document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing document metadata
        """
        metadata = {
            'filename': Path(file_path).name,
            'file_size': os.path.getsize(file_path),
            'file_type': Path(file_path).suffix.lower()
        }
        
        return metadata


def parse_policy_document(file_path: str, chunk_size: int = 500) -> List[str]:
    """
    Convenience function to parse a policy document.
    
    Args:
        file_path: Path to the policy document
        chunk_size: Target tokens per chunk
        
    Returns:
        List of text chunks
    """
    parser = DocumentParser(chunk_size=chunk_size)
    return parser.parse_document(file_path)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    parser = DocumentParser()
    
    # Test with a sample file (when available)
    try:
        sample_file = "data/sample_policies/sample_policy.pdf"
        if os.path.exists(sample_file):
            chunks = parser.parse_document(sample_file)
            print(f"Parsed {len(chunks)} chunks from {sample_file}")
            
            # Display first chunk as example
            if chunks:
                print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")
        else:
            print("No sample file found for testing")
            
    except Exception as e:
        print(f"Error during testing: {e}") 