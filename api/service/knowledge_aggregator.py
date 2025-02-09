from typing import List, Dict
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor
import json

class KnowledgeAggregator:
    def __init__(self):
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(api_key="SHHHNXH43KX5AF986LR7WEPFFILV6OKKKIT4NVSF", model="cognitivecomputations/dolphin-2.9-llama3-8b", base_url="https://bbiopuow45en08-8000.proxy.runpod.net/v1", temperature=0.7)
        
        # Initialize search tools
        self.search = DuckDuckGoSearchRun()
        
        # Create tools list
        self.tools = [
            Tool(
                name="Web Search",
                func=self.search.run,
                description="Useful for searching information about companies and their CSR activities"
            )
        ]
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # Initialize competitor analysis prompt
        self.competitor_prompt = PromptTemplate(
            input_variables=["company", "industry"],
            template="""
            Find the top competitors of {company} in the {industry} industry who are known for their CSR activities.
            Focus on:
            1. Environmental initiatives
            2. Social responsibility programs
            3. Community engagement
            4. Sustainability practices
            
            Return the information in a structured format.
            """
        )
        
        # Initialize CSR analysis prompt
        self.csr_prompt = PromptTemplate(
            input_variables=["company"],
            template="""
            Find detailed information about CSR activities and social awareness initiatives by {company}.
            Focus on:
            1. Recent hackathons or tech events
            2. Environmental programs
            3. Educational initiatives
            4. Community development projects
            
            Return the information in a structured format.
            """
        )
        
        # Initialize trend analysis prompt
        self.trend_prompt = PromptTemplate(
            input_variables=["industry", "competitor_data"],
            template="""
            Based on the following competitor CSR activities and industry context, identify current trending and potentially attractive CSR activities.
            
            Industry: {industry}
            Competitor Activities: {competitor_data}
            
            Please analyze the data and suggest innovative CSR activities that would be attractive and feasible.
            Focus on:
            1. Current trends in CSR
            2. Activities that have shown high engagement
            3. Innovative approaches that stand out
            4. Activities that align with current social and environmental concerns
            
            Return the suggestions in this exact JSON array format:
            [
                {{"name": "Activity Name", "description": "Detailed description of the activity and its potential impact"}},
                // more activities...
            ]
            """
        )
    
    def analyze_competitor(self, company_name: str, industry: str) -> Dict:
        """Analyze a competitor's CSR activities"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.competitor_prompt)
            result = chain.run(company=company_name, industry=industry)
            return {"company": company_name, "analysis": result}
        except Exception as e:
            return {"company": company_name, "error": str(e)}
    
    def analyze_csr_activities(self, company_name: str) -> Dict:
        """Analyze specific CSR activities of a company"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.csr_prompt)
            result = chain.run(company=company_name)
            return {"company": company_name, "csr_activities": result}
        except Exception as e:
            return {"company": company_name, "error": str(e)}
    
    def recursive_competitor_analysis(self, company: str, industry: str, depth: int = 2) -> Dict:
        """Recursively analyze competitors and their CSR activities"""
        if depth <= 0:
            return {}
        
        # Analyze initial company
        results = {"company": company}
        competitor_data = self.analyze_competitor(company, industry)
        
        # Extract competitor names using the LLM
        competitor_extraction_prompt = PromptTemplate(
            input_variables=["analysis"],
            template="Extract the company names mentioned in this text: {analysis}"
        )
        chain = LLMChain(llm=self.llm, prompt=competitor_extraction_prompt)
        competitors_text = chain.run(analysis=competitor_data["analysis"])
        
        # Process competitors recursively
        competitors = [comp.strip() for comp in competitors_text.split(",") if comp.strip()][:3]
        competitor_results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_company = {
                executor.submit(self.recursive_competitor_analysis, comp, industry, depth - 1): comp 
                for comp in competitors
            }
            for future in future_to_company:
                competitor_results.append(future.result())
        
        results["analysis"] = competitor_data["analysis"]
        results["competitors"] = competitor_results
        return results
    
    def analyze_trends_and_suggest_activities(self, industry: str, competitor_data: Dict) -> List[Dict]:
        """Analyze trends and suggest CSR activities based on competitor analysis"""
        try:
            # Convert competitor data to a more readable format for the LLM
            competitor_summary = json.dumps(competitor_data, indent=2)
            
            # Run the trend analysis
            chain = LLMChain(llm=self.llm, prompt=self.trend_prompt)
            result = chain.run(industry=industry, competitor_data=competitor_summary)
            
            # Parse the result into a list of dictionaries
            try:
                suggestions = json.loads(result)
                return suggestions
            except json.JSONDecodeError:
                # Fallback in case the LLM doesn't return valid JSON
                return [{"name": "Error", "description": "Failed to parse suggestions"}]
                
        except Exception as e:
            return [{"name": "Error", "description": f"Failed to analyze trends: {str(e)}"}]

    def aggregate_knowledge(self, company: str, industry: str) -> Dict:
        """Main method to aggregate knowledge about a company and its competitors"""
        try:
            # Initial analysis
            competitor_tree = self.recursive_competitor_analysis(company, industry)
            
            # Analyze CSR activities for the main company
            csr_activities = self.analyze_csr_activities(company)
            
            # Analyze trends and suggest activities
            suggested_activities = self.analyze_trends_and_suggest_activities(
                industry,
                {
                    "competitor_analysis": competitor_tree,
                    "current_csr": csr_activities
                }
            )
            
            # Combine results
            return {
                "company": company,
                "industry": industry,
                "csr_activities": csr_activities,
                "competitor_analysis": competitor_tree,
                "suggested_activities": suggested_activities
            }
        except Exception as e:
            return {"error": str(e)} 