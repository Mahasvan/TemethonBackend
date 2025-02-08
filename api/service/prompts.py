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

Focus on the most important resources for the given company - for example an IT company would have it's most important resources as energy, construction - land resource, food industry - water resource, etc. etc. Dont halluinate. Looking for top level natural resources dont go into detail on extremely specific resources - rather classifications of resources in a general manner.
Return only the JSON array without any additional text or explanation.""",
    input_variables=["description"]
)

REPORT_TEMPLATE_PROMPT = PromptTemplate(
    template="""You are a bot that is designed to generate a format for a very detailed report for an organization, for its ESG and CSR reports.
    You will be given a list of sustainability initiatives that the company has undertaken. 
    You will also be given data about various metrics of the ESG parameters. 
    Your job is to make a structure for the document, in the following format:
    
    For each ESG Metric or CSR Initiative, give the following structure enclosed within angle braces <>:
    < 
    This is a report on CSR Initiative X, which has the following fields:
    - start date: XXXX-MM-DD etc,
    - end date: XXXX-MM-DD etc,
    and whatever other details is given for each initiative, or metric.
     >
     
     You also need to provide visualizations in between, detailing the kind of data you want visualised.
     <
     Visualization - Gantt chart showing the CSR initiatives over time 
     >
     < 
     Visualization showing the trend of ESG Metrics, compared with the goals set
     >
     etc.
     
     Here is the data.
     {data}
    """,
    input_variables=["data"]
)