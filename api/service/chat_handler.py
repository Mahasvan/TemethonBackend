import json
import os
import pprint

from langchain.chains import LLMChain
from langchain_groq import ChatGroq

from api.service.prompts import (INDUSTRY_CLASSIFICATION_PROMPT,
                                 INITIATIVE_CLASSIFICATION_PROMPT, MATERIALITY_ASSESSMENT_PROMPT,
                                 REPORT_TEMPLATE_PROMPT)

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
)

industry_classification_chain = LLMChain(
    llm=llm,
    prompt=INDUSTRY_CLASSIFICATION_PROMPT
)

initiative_classification_chain = LLMChain(
    llm=llm,
    prompt=INITIATIVE_CLASSIFICATION_PROMPT
)

materiality_assessment_chain = LLMChain(
    llm=llm,
    prompt=MATERIALITY_ASSESSMENT_PROMPT
)

report_structure_chain = LLMChain(
    llm=llm,
    prompt=REPORT_TEMPLATE_PROMPT
)
# industry_classification_chain = INDUSTRY_CLASSIFICATION_PROMPT | llm

# initiative_classification_chain = INITIATIVE_CLASSIFICATION_PROMPT | llm