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
from scipy import stats
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

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

class ScoreInput(BaseModel):
    score: float
    mode: str

class StockDataInput(BaseModel):
    symbol: str

class VolatilityInput(BaseModel):
    symbol: str

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
    
    # Replace the pipe operator chain with explicit chain creation
    messages = prompt.format_messages(
        text=chunk.page_content,
        format_instructions=format_instructions
    )
    
    try:
        logger.info(f"Sending chunk {chunk_index + 1} to LLM for analysis")
        response = llm.invoke(messages)
        result = parser.parse(response.content)
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

@router.post("/percentile")
async def calculate_percentile(input_data: ScoreInput):
    """
    Calculate the percentile and Gaussian distribution parameters for a given ESG score.
    
    Args:
        input_data: ScoreInput containing score and mode (env, soc, gov, or total)
    
    Returns:
        JSON response with percentile and Gaussian parameters
    """
    # Population statistics
    models = {
        "env": {
            "mean": 48,
            "max": 80,
            "min": 7,
        },
        "soc": {
            "mean": 57,
            "max": 78,
            "min": 35,
        },
        "gov": {
            "mean": 75,
            "max": 84,
            "min": 43,
        },
        "total": {
            "mean": 64,
            "max": 80,
            "min": 40,
        }
    }
    
    if input_data.mode not in models:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be one of: env, soc, gov, total")
    
    if not 1 <= input_data.score <= 100:
        raise HTTPException(status_code=400, detail="Score must be between 1 and 100")
    
    model = models[input_data.mode]
    
    # Calculate standard deviation using the empirical rule
    std_dev = (model["max"] - model["min"]) / 4
    
    # Calculate percentile using normal distribution
    percentile = stats.norm.cdf(input_data.score, loc=model["mean"], scale=std_dev) * 100
    
    # Calculate Gaussian parameters
    gaussian_params = {
        "mean": model["mean"],
        "std_dev": std_dev,
        "variance": std_dev ** 2
    }
    
    return JSONResponse({
        "status": "success",
        "data": {
            "score": input_data.score,
            "mode": input_data.mode,
            "percentile": round(percentile, 2),
            "population_stats": model,
            "gaussian_parameters": gaussian_params
        }
    })

@router.post("/nasdaq-historical")
async def get_nasdaq_historical(input_data: StockDataInput):
    """
    Download historical stock data from NASDAQ using Yahoo Finance.
    Default period is last 30 days up to today.
    
    Args:
        input_data: StockDataInput containing symbol
        
    Returns:
        JSON response with historical stock data
    """
    try:
        # Always get data for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Download data using yfinance
        ticker = yf.Ticker(input_data.symbol)
        df = ticker.history(period="30d")  # Use period parameter instead of start/end dates
        
        # Process the data
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Convert to dictionary format
        historical_data = df.to_dict('records')
        
        # Get company info
        info = ticker.info
        company_info = {
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency")
        }
        
        return JSONResponse({
            "status": "success",
            "data": {
                "symbol": input_data.symbol,
                "company_info": company_info,
                "historical_data": historical_data,
                "period": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching NASDAQ data: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching historical data: {str(e)}"
        )

@router.post("/volatility")
async def analyze_volatility(input_data: VolatilityInput):
    """
    Analyze stock price volatility by comparing prices against a trend line.
    Also analyzes how variance changes over time to indicate customer satisfaction trends.
    Uses 30 days of historical data.
    """
    try:
        # Get historical data
        ticker = yf.Ticker(input_data.symbol)
        df = ticker.history(period="30d")
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No data available for this symbol"
            )
            
        if len(df) < 5:  # Need at least 5 days of data for meaningful analysis
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for analysis. Need at least 5 days of trading data."
            )
        
        # Reset index and ensure we have dates as column
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day'] = (df['Date'] - df['Date'].min()).dt.days
        
        # Calculate price trend line
        price_regression = stats.linregress(df['Day'], df['Close'])
        df['Price_Trend'] = price_regression.slope * df['Day'] + price_regression.intercept
        
        # Calculate variance metrics with safety checks
        df['Variance'] = df['Close'] - df['Price_Trend']
        df['Variance_Pct'] = df['Variance'] / df['Price_Trend'].replace(0, float('nan')) * 100
        df['Variance_Pct'] = df['Variance_Pct'].fillna(0)  # Replace NaN with 0
        
        # Calculate rolling variance (5-day window) to smooth out daily fluctuations
        df['Rolling_Variance'] = df['Variance_Pct'].rolling(window=5, min_periods=1).std()
        df['Rolling_Variance'] = df['Rolling_Variance'].fillna(0)  # Replace NaN with 0
        
        # Analyze trend in variance itself
        variance_regression = stats.linregress(df['Day'], df['Rolling_Variance'])
        df['Variance_Trend'] = variance_regression.slope * df['Day'] + variance_regression.intercept
        
        def safe_float(value):
            """Convert value to float, replacing inf/nan with 0"""
            try:
                if pd.isna(value) or pd.isinf(value):
                    return 0.0
                return float(value)
            except:
                return 0.0
        
        # Calculate volatility metrics with safety checks
        volatility_metrics = {
            "current_metrics": {
                "mean_variance": safe_float(df['Variance'].mean()),
                "std_variance": safe_float(df['Variance'].std()),
                "max_variance": safe_float(df['Variance'].max()),
                "min_variance": safe_float(df['Variance'].min()),
                "mean_variance_pct": safe_float(df['Variance_Pct'].mean()),
                "std_variance_pct": safe_float(df['Variance_Pct'].std()),
            },
            "trend_analysis": {
                "price_trend": {
                    "slope": safe_float(price_regression.slope),
                    "r_squared": safe_float(price_regression.rvalue ** 2),
                },
                "variance_trend": {
                    "slope": safe_float(variance_regression.slope),
                    "r_squared": safe_float(variance_regression.rvalue ** 2),
                    "direction": "increasing" if variance_regression.slope > 0 else "decreasing",
                    "significance": get_trend_significance(variance_regression.pvalue)
                }
            },
            "customer_satisfaction_indicator": get_satisfaction_indicator(
                safe_float(variance_regression.slope),
                safe_float(df['Rolling_Variance'].mean())
            )
        }
        
        # Prepare daily data for response, ensuring all values are JSON-serializable
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        daily_data = []
        for record in df[[
            'Date', 'Close', 'Price_Trend', 'Variance_Pct', 
            'Rolling_Variance', 'Variance_Trend'
        ]].to_dict('records'):
            safe_record = {k: safe_float(v) if k != 'Date' else v for k, v in record.items()}
            daily_data.append(safe_record)
        
        return JSONResponse({
            "status": "success",
            "data": {
                "symbol": input_data.symbol,
                "volatility_metrics": volatility_metrics,
                "daily_data": daily_data,
                "summary": {
                    "current_volatility": get_volatility_rating(
                        safe_float(volatility_metrics['current_metrics']['std_variance_pct'])
                    ),
                    "volatility_trend": get_volatility_trend_description(
                        safe_float(variance_regression.slope),
                        variance_regression.pvalue
                    ),
                    "customer_satisfaction_trend": get_satisfaction_trend(
                        safe_float(variance_regression.slope),
                        variance_regression.pvalue
                    )
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing volatility: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing volatility: {str(e)}"
        )

def get_trend_significance(p_value: float) -> str:
    """
    Determine the statistical significance of a trend.
    """
    if p_value < 0.01:
        return "Very Strong"
    elif p_value < 0.05:
        return "Strong"
    elif p_value < 0.1:
        return "Moderate"
    else:
        return "Weak"

def get_satisfaction_indicator(variance_slope: float, mean_variance: float) -> str:
    """
    Interpret variance trends in terms of customer satisfaction.
    """
    if variance_slope < -0.01:
        return "Improving - Volatility is decreasing, suggesting increasing customer satisfaction"
    elif variance_slope > 0.01:
        return "Declining - Increasing volatility suggests potential customer satisfaction issues"
    else:
        if mean_variance < 2.0:
            return "Stable - Low volatility indicates consistent customer satisfaction"
        else:
            return "Unstable - High but steady volatility suggests ongoing satisfaction challenges"

def get_volatility_trend_description(slope: float, p_value: float) -> str:
    """
    Provide a description of how volatility is changing over time.
    """
    significance = get_trend_significance(p_value)
    if abs(slope) < 0.001:
        return f"Volatility is stable ({significance} confidence)"
    
    direction = "increasing" if slope > 0 else "decreasing"
    magnitude = "rapidly" if abs(slope) > 0.05 else "gradually"
    
    return f"Volatility is {magnitude} {direction} ({significance} confidence)"

def get_satisfaction_trend(slope: float, p_value: float) -> str:
    """
    Interpret volatility trends in terms of customer satisfaction changes.
    """
    if p_value >= 0.1:
        return "No clear trend in customer satisfaction"
    
    if slope > 0.01:
        return "Customer satisfaction appears to be declining"
    elif slope < -0.01:
        return "Customer satisfaction shows signs of improvement"
    else:
        return "Customer satisfaction remains stable"

def get_volatility_rating(std_variance_pct: float) -> str:
    """
    Convert standard deviation of variance percentage into a rating.
    """
    if std_variance_pct < 1.0:
        return "Very Low"
    elif std_variance_pct < 2.0:
        return "Low"
    elif std_variance_pct < 3.0:
        return "Moderate"
    elif std_variance_pct < 4.0:
        return "High"
    else:
        return "Very High"

def setup(app):
    """Setup function to register the router with the FastAPI app"""
    try:
        app.include_router(router)
        logger.info("ESG router successfully registered")
        return True
    except Exception as e:
        logger.error(f"Failed to setup ESG router: {str(e)}")
        return False