from langchain.prompts import PromptTemplate

INITIATIVE_CLASSIFICATION_PROMPT = PromptTemplate(
    template="""You are a CSR (Corporate Social Responsibility) initiative analyzer. Analyze the following event description and extract the required information.

The tracks available are:
- Environmental sustainability
- Community Development and Social Impact
- Employee well being and workplace ethics
- Ethical business practice and governance
- Philanthropy and charitable giving

The current month and year are February and 2025.

IMPORTANT REQUIREMENTS:
1. ALL of these fields MUST be present and non-null for a complete response:
   - name
   - description
   - start_date (YYYY-MM-DD format)
   - end_date (YYYY-MM-DD format)
   - track (must be one from the list above)
   - metrics (at least 2 meaningful metrics)

2. For metrics, each metric MUST be a list of THREE strings: [quantity, unit, description]
3. If the determined track relates to trees / environment - it is mandatory to ask how many trees were planted and return that in the metric. 
If ANY of the required fields are missing or null, you MUST:
1. Set "complete" to false
2. List specific questions in the "questions" field to gather the missing information
3. Do not make assumptions about missing data

Description: {description}

Return a JSON object with the following structure:
{{
    "name": "event name",
    "description": "brief description",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "track": "track name",
    "metrics": [["quantity1", "unit1", "description1"], ["quantity2", "unit2", "description2"]],
    "complete": boolean,
    "questions": ["question1", "question2"] or null if complete
}}

Return only the JSON output without any additional text or explanation.
""",
    input_variables=["description"]
)

INDUSTRY_CLASSIFICATION_PROMPT = PromptTemplate(
    template=""""Given the following company mission statement and description, classify the company into one of the following industries and provide your response as a single integer:

0: Biofuels
1: Biotechnology & Pharmaceuticals
2: Software & IT Services
3: Food Retailers & Distributors
4: Oil & Gas – Exploration & Production
Company Mission Statement and Description:

Return only the integer output without any additional text or explanation. If it is Biofuels, return only "0".
Mission statement and Description:
{description}
""", input_variables=["description"]
)

MATERIALITY_ASSESSMENT_PROMPT = PromptTemplate(
    template="""Given the following company's mission statement, vision statement, and industry, identify the most impactful Environmental, Social, and Governance (ESG) issues relevant to the business. These issues should be ordered by importance and include detailed descriptions.

Company Description:
{description}

Return a JSON array containing objects with the following structure, ordered by importance:
"name": "Issue Name",
"desc": "Detailed description of why this issue is material to the company"

Focus on industry-specific material factors that have significant impact on operations, sustainability, and long-term value creation. Include both opportunities and risks.

Return only the JSON array without any additional text or explanation.""",
    input_variables=["description"]
)