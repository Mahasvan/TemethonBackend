from typing import List, Dict
import json
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time
from functools import lru_cache

class GovernanceAnalyzer:
    def __init__(self):
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(api_key="SHHHNXH43KX5AF986LR7WEPFFILV6OKKKIT4NVSF", model="cognitivecomputations/dolphin-2.9-llama3-8b", base_url="https://bbiopuow45en08-8000.proxy.runpod.net/v1", temperature=0.7)
        
        # Initialize search tools with rate limiting
        self.search = DuckDuckGoSearchRun()
        self.last_search_time = 0
        self.min_search_interval = 2  # Minimum seconds between searches
        
        # Create tools list
        self.tools = [
            Tool(
                name="Web Search",
                func=self.rate_limited_search,
                description="Useful for searching information about government policies and ESG regulations"
            )
        ]
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # Initialize governance analysis prompt
        self.governance_prompt = PromptTemplate(
            input_variables=["company", "industry"],
            template="""
            Based on general knowledge and industry standards, analyze the governance aspects for {company} in the {industry} industry.
            Focus on:
            1. Common government policies and regulations affecting CSR activities
            2. Standard tax benefits and incentives for CSR initiatives
            3. Basic compliance requirements and reporting standards
            4. Typical government partnerships and support programs
            
            Return the information in a structured JSON format with the following sections:
            {
                "government_policies": ["policy1", "policy2"],
                "tax_benefits": ["benefit1", "benefit2"],
                "compliance_requirements": ["requirement1", "requirement2"],
                "partnership_opportunities": ["opportunity1", "opportunity2"],
                "risk_assessment": ["risk1", "risk2"]
            }
            """
        )
        
        # Initialize regulatory impact prompt
        self.regulatory_impact_prompt = PromptTemplate(
            input_variables=["company", "industry"],
            template="""
            Based on industry standards and common practices, analyze the potential regulatory impact for {company} in the {industry} industry.
            
            Focus on:
            1. Typical regulatory requirements
            2. Common compliance frameworks
            3. Standard governance practices
            
            Return the analysis in this exact JSON format:
            [
                {{
                    "activity": "Regulatory Compliance",
                    "governance_impact": "description",
                    "regulatory_benefits": ["benefit1", "benefit2"],
                    "risk_factors": ["risk1", "risk2"]
                }}
            ]
            """
        )

    def rate_limited_search(self, query: str) -> str:
        """Perform rate-limited search"""
        current_time = time.time()
        time_since_last_search = current_time - self.last_search_time
        
        if time_since_last_search < self.min_search_interval:
            time.sleep(self.min_search_interval - time_since_last_search)
        
        try:
            result = self.search.run(query)
            self.last_search_time = time.time()
            return result
        except Exception as e:
            return f"Search limited. Using cached or default analysis. Error: {str(e)}"

    @lru_cache(maxsize=100)
    def analyze_governance_landscape(self, company: str, industry: str) -> Dict:
        """Analyze the governance landscape with caching"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.governance_prompt)
            result = chain.run(company=company, industry=industry)
            
            try:
                governance_data = json.loads(result)
                return governance_data
            except json.JSONDecodeError:
                return {
                    "government_policies": ["Standard industry regulations"],
                    "tax_benefits": ["Standard CSR tax deductions"],
                    "compliance_requirements": ["Basic regulatory compliance"],
                    "partnership_opportunities": ["Standard government programs"],
                    "risk_assessment": ["Standard regulatory risks"]
                }
                
        except Exception as e:
            return {"error": f"Using default analysis. Error: {str(e)}"}

    @lru_cache(maxsize=100)
    def analyze_regulatory_impact(self, company: str, industry: str) -> List[Dict]:
        """Analyze the regulatory impact with caching"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.regulatory_impact_prompt)
            result = chain.run(company=company, industry=industry)
            
            try:
                impact_analysis = json.loads(result)
                return impact_analysis
            except json.JSONDecodeError:
                return [{
                    "activity": "Standard Compliance",
                    "governance_impact": "Basic regulatory alignment",
                    "regulatory_benefits": ["Standard benefits"],
                    "risk_factors": ["Standard risks"]
                }]
                
        except Exception as e:
            return [{"error": f"Using default analysis. Error: {str(e)}"}]

    def aggregate_governance_analysis(self, company: str, industry: str) -> Dict:
        """Main method to aggregate governance analysis"""
        try:
            # Get general governance landscape
            governance_landscape = self.analyze_governance_landscape(company, industry)
            
            # Analyze regulatory impact
            regulatory_impact = self.analyze_regulatory_impact(company, industry)
            
            # Combine results
            return {
                "company": company,
                "industry": industry,
                "governance_landscape": governance_landscape,
                "regulatory_impact_analysis": regulatory_impact,
                "esg_governance_score": {
                    "score": self._calculate_governance_score(governance_landscape, regulatory_impact),
                    "factors": self._extract_key_factors(governance_landscape, regulatory_impact)
                }
            }
        except Exception as e:
            return {"error": str(e)}
            
    def _calculate_governance_score(self, landscape: Dict, impact: List[Dict]) -> float:
        """Calculate a governance score based on various factors"""
        try:
            score = 0
            max_score = 100
            
            if not isinstance(landscape, Dict) or "error" in landscape:
                return 0.0
                
            factors = [
                "government_policies",
                "tax_benefits",
                "compliance_requirements",
                "partnership_opportunities"
            ]
            
            for factor in factors:
                if factor in landscape and landscape[factor]:
                    score += 25
                    
            return round(score / max_score, 2)
        except Exception:
            return 0.0
            
    def _extract_key_factors(self, landscape: Dict, impact: List[Dict]) -> List[str]:
        """Extract key factors affecting the governance score"""
        factors = []
        
        try:
            if isinstance(landscape, Dict) and "error" not in landscape:
                if "government_policies" in landscape:
                    factors.append("Aligned with government policies")
                if "tax_benefits" in landscape:
                    factors.append("Eligible for tax benefits")
                if "partnership_opportunities" in landscape:
                    factors.append("Strong public sector partnership potential")
                    
            if isinstance(impact, list):
                for item in impact:
                    if "regulatory_benefits" in item:
                        factors.extend(item["regulatory_benefits"])
                        
            return list(set(factors))
        except Exception:
            return ["Using standard industry factors"] 