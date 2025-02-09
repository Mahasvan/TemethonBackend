from typing import Dict, List
import json
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time
from functools import lru_cache

class Calculator:
    """Simple calculator tool for the agent"""
    def add(self, a: float, b: float) -> float:
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class EmissionsCalculator:
    def __init__(self):
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(api_key="SHHHNXH43KX5AF986LR7WEPFFILV6OKKKIT4NVSF", model="cognitivecomputations/dolphin-2.9-llama3-8b", base_url="https://bbiopuow45en08-8000.proxy.runpod.net/v1", temperature=0.7)
        
        # Initialize calculator
        self.calculator = Calculator()
        
        # Initialize search tools with rate limiting
        self.search = DuckDuckGoSearchRun()
        self.last_search_time = 0
        self.min_search_interval = 2
        
        # Create tools list
        self.tools = [
            Tool(
                name="Search",
                func=self.rate_limited_search,
                description="Useful for finding information about energy mix, emissions factors, and environmental data"
            ),
            Tool(
                name="Calculator",
                func=self.calculate,
                description="Useful for performing mathematical calculations. Input should be in format: operation,num1,num2 (e.g., 'multiply,5,3' or 'add,10,20')"
            )
        ]
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # Initialize energy mix analysis prompt
        self.energy_mix_prompt = PromptTemplate(
            input_variables=["location"],
            template="""
            Find the energy mix distribution and emissions factors for {location}.
            Focus on:
            1. Percentage distribution of energy sources (coal, gas, solar, wind, hydro, nuclear, etc.)
            2. CO2 emissions factor for each energy source (in kg CO2/kWh)
            
            Return the information in this JSON format:
            {{
                "energy_mix": {{
                    "coal": percentage,
                    "gas": percentage,
                    "solar": percentage,
                    "wind": percentage,
                    "hydro": percentage,
                    "nuclear": percentage,
                    "other": percentage
                }},
                "emissions_factors": {{
                    "coal": number,
                    "gas": number,
                    "solar": number,
                    "wind": number,
                    "hydro": number,
                    "nuclear": number,
                    "other": number
                }}
            }}
            """
        )
        
        # Initialize emissions calculation prompt
        self.emissions_prompt = PromptTemplate(
            input_variables=["location", "data", "energy_mix"],
            template="""
            Calculate the total CO2 emissions in metric tonnes for a company based on the following data:
            Location: {location}
            Data: {data}
            Energy Mix Data: {energy_mix}
            
            Follow these steps:
            1. Calculate electricity emissions:
               - Distribute total kWh according to energy mix percentages
               - Multiply each portion by its emissions factor
               - Sum up total emissions from electricity
               - Convert to metric tonnes
            2. Calculate commute emissions:
               - Find average CO2 emissions per km for commuting
               - Multiply by number of in-office employees (total - WFH)
            3. Calculate WFH emissions:
               - Find average CO2 emissions for WFH setup
               - Multiply by WFH employee count
            4. Add transmission and distribution losses
            5. Add waste emissions
            
            Return the calculations in this JSON format:
            {{
                "electricity_emissions": {{
                    "by_source": {{
                        "coal": number,
                        "gas": number,
                        "solar": number,
                        "wind": number,
                        "hydro": number,
                        "nuclear": number,
                        "other": number
                    }},
                    "total": number
                }},
                "commute_emissions": number,
                "wfh_emissions": number,
                "td_losses": number,
                "waste_emissions": number,
                "total_emissions": number,
                "calculation_breakdown": {{
                    "energy_consumption_kwh": number,
                    "average_commute_distance": number,
                    "emissions_per_km": number,
                    "wfh_emissions_per_employee": number
                }}
            }}
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
            return f"Search limited. Using default values. Error: {str(e)}"

    def calculate(self, operation: str) -> float:
        """Handle calculator operations"""
        try:
            op, num1, num2 = operation.split(',')
            num1, num2 = float(num1), float(num2)
            
            if op == 'add':
                return self.calculator.add(num1, num2)
            elif op == 'subtract':
                return self.calculator.subtract(num1, num2)
            elif op == 'multiply':
                return self.calculator.multiply(num1, num2)
            elif op == 'divide':
                return self.calculator.divide(num1, num2)
            else:
                raise ValueError(f"Unknown operation: {op}")
        except Exception as e:
            return f"Calculation error: {str(e)}"

    @lru_cache(maxsize=100)
    def get_energy_mix(self, location: str) -> Dict:
        """Get energy mix and emissions factors for a location"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.energy_mix_prompt)
            result = chain.run(location=location)
            
            try:
                energy_data = json.loads(result)
                return energy_data
            except json.JSONDecodeError:
                return self.get_default_energy_mix()
                
        except Exception as e:
            return self.get_default_energy_mix()

    def calculate_emissions(self, data: Dict) -> Dict:
        """Calculate total CO2 emissions based on provided data"""
        try:
            # Get energy mix data
            energy_mix_data = self.get_energy_mix(data["location"])
            
            # Format data for the prompt
            data_str = json.dumps(data, indent=2)
            energy_mix_str = json.dumps(energy_mix_data, indent=2)
            
            # Create chain for emissions calculation
            chain = LLMChain(llm=self.llm, prompt=self.emissions_prompt)
            
            # Run the calculation
            result = chain.run(
                location=data["location"],
                data=data_str,
                energy_mix=energy_mix_str
            )
            
            try:
                emissions_data = json.loads(result)
                return emissions_data
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse emissions calculation",
                    "raw_result": result
                }
                
        except Exception as e:
            return {"error": f"Failed to calculate emissions: {str(e)}"}

    def get_default_energy_mix(self) -> Dict:
        """Get default energy mix when search fails"""
        return {
            "energy_mix": {
                "coal": 30,
                "gas": 20,
                "solar": 10,
                "wind": 10,
                "hydro": 15,
                "nuclear": 10,
                "other": 5
            },
            "emissions_factors": {
                "coal": 0.9,  # kg CO2/kWh
                "gas": 0.37,
                "solar": 0.048,
                "wind": 0.011,
                "hydro": 0.024,
                "nuclear": 0.012,
                "other": 0.5
            }
        } 