from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import PyPDF2
import io
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List, Dict
import json
from langchain_core.documents import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter(
    prefix="/esg",
    tags=["ESG Analysis"]
)

class PDFUrlInput(BaseModel):
    pdf_url: str

def download_and_extract_pdf_text(pdf_url: str) -> str:
    try:
        logger.info(f"Downloading PDF from URL: {pdf_url}")
        # Download PDF from URL
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Read PDF content
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        total_pages = len(pdf_reader.pages)
        logger.info(f"Processing PDF with {total_pages} pages")
        
        for i, page in enumerate(pdf_reader.pages, 1):
            logger.info(f"Extracting text from page {i}/{total_pages}")
            text += page.extract_text() + "\n"
        
        logger.info(f"Successfully extracted {len(text)} characters of text")
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def split_text_into_chunks(text: str) -> List[Document]:
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split text into chunks
    chunks = text_splitter.create_documents([text])
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def analyze_esg_chunk(chunk: Document, llm: ChatOpenAI, chunk_index: int) -> dict:
    logger.info(f"Processing chunk {chunk_index + 1} with {len(chunk.page_content)} characters")
    
    # Define output schemas for chunk analysis
    chunk_schema = ResponseSchema(
        name="esg_elements",
        description="""JSON object containing identified ESG elements with the following structure:
        {
            "environmental": ["list of environmental elements found"],
            "social": ["list of social elements found"],
            "governance": ["list of governance elements found"]
        }"""
    )
    
    parser = StructuredOutputParser.from_response_schemas([chunk_schema])
    format_instructions = parser.get_format_instructions()

    template = """Analyze this section of a company document and identify any ESG (Environmental, Social, and Governance) related elements. Focus on:

    Environmental:
    - Carbon emissions and energy
    - Water and waste management
    - Climate risk and sustainability

    Social:
    - Diversity and labor practices
    - Health, safety, and community
    - Customer and data matters

    Governance:
    - Board and compensation
    - Shareholder rights
    - Ethics and compliance

    Document section: {text}

    {format_instructions}

    Extract and categorize all ESG-related information found in this section. Return ONLY the JSON object with the findings."""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | parser

    try:
        logger.info(f"Sending chunk {chunk_index + 1} to LLM for analysis")
        result = chain.invoke({
            "text": chunk.page_content,
            "format_instructions": format_instructions
        })
        logger.info(f"Successfully processed chunk {chunk_index + 1}")
        return result["esg_elements"]
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index + 1}: {str(e)}")
        logger.error(f"Chunk content preview: {chunk.page_content[:200]}...")
        return {
            "environmental": [],
            "social": [],
            "governance": []
        }

def calculate_esg_scores(analyses: List[Dict]) -> dict:
    logger.info(f"Calculating final scores from {len(analyses)} chunk analyses")
    
    # Initialize counters for each category
    env_elements = []
    social_elements = []
    gov_elements = []
    
    # Collect elements from all chunks
    for i, analysis in enumerate(analyses):
        logger.info(f"Processing analysis results from chunk {i + 1}")
        if isinstance(analysis, dict):
            env_elements.extend(analysis.get('environmental', []))
            social_elements.extend(analysis.get('social', []))
            gov_elements.extend(analysis.get('governance', []))
    
    # Remove duplicates
    env_elements = list(set(env_elements))
    social_elements = list(set(social_elements))
    gov_elements = list(set(gov_elements))
    
    logger.info(f"Found {len(env_elements)} environmental, {len(social_elements)} social, and {len(gov_elements)} governance elements")
    
    # Calculate scores based on the number and quality of elements found
    def calculate_category_score(elements: list, max_elements: int = 10) -> float:
        return min(100, (len(elements) / max_elements) * 100)
    
    env_score = calculate_category_score(env_elements)
    social_score = calculate_category_score(social_elements)
    gov_score = calculate_category_score(gov_elements)
    
    # Calculate final weighted score
    final_score = (
        env_score * 0.4 +    # 40% weight
        social_score * 0.35 + # 35% weight
        gov_score * 0.25     # 25% weight
    )
    
    # Prepare explanation
    explanation = {
        "environmental_elements": env_elements,
        "social_elements": social_elements,
        "governance_elements": gov_elements,
        "methodology": "Scores are calculated based on the presence and quality of ESG elements in the document. Each category is evaluated independently and then weighted for the final score.",
        "total_elements_found": {
            "environmental": len(env_elements),
            "social": len(social_elements),
            "governance": len(gov_elements)
        }
    }
    
    logger.info(f"Final ESG scores calculated - Environmental: {env_score:.2f}, Social: {social_score:.2f}, Governance: {gov_score:.2f}, Final: {final_score:.2f}")
    
    return {
        "environmental_score": round(env_score, 2),
        "social_score": round(social_score, 2),
        "governance_score": round(gov_score, 2),
        "final_score": round(final_score, 2),
        "explanation": explanation
    }

@router.post("/analyze")
async def analyze_esg_from_pdf(input_data: PDFUrlInput):
    """
    Analyze ESG metrics from a company's code of conduct PDF document.
    Provides scores for environmental, social, and governance factors.
    """
    logger.info("Starting ESG analysis for PDF")
    
    # Extract text from PDF
    text = download_and_extract_pdf_text(input_data.pdf_url)
    
    # Split text into manageable chunks
    chunks = split_text_into_chunks(text)
    
    # Initialize LLM
    try:
        logger.info("Initializing LLM")
        llm = ChatOpenAI(api_key="SHHHNXH43KX5AF986LR7WEPFFILV6OKKKIT4NVSF", model="cognitivecomputations/dolphin-2.9-llama3-8b", base_url="https://bbiopuow45en08-8000.proxy.runpod.net/v1", temperature=0.4)

    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing language model: {str(e)}")
    
    # Analyze each chunk
    chunk_analyses = []
    for i, chunk in enumerate(chunks):
        try:
            analysis = analyze_esg_chunk(chunk, llm, i)
            if analysis:
                chunk_analyses.append(analysis)
        except Exception as e:
            logger.error(f"Error in chunk {i + 1} analysis: {str(e)}")
    
    if not chunk_analyses:
        logger.error("No successful chunk analyses completed")
        raise HTTPException(status_code=500, detail="Failed to analyze document content")
    
    # Calculate final scores
    try:
        result = calculate_esg_scores(chunk_analyses)
        logger.info("ESG analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error calculating final scores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating ESG scores: {str(e)}")

def setup(app):
    """Setup function to register the router with the FastAPI app"""
    try:
        app.include_router(router)
        logger.info("ESG router successfully registered")
        return True
    except Exception as e:
        logger.error(f"Failed to setup ESG router: {str(e)}")
        return False 