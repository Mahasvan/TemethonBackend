import json
import os
import pprint

from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

from api.service.prompts import INDUSTRY_CLASSIFICATION_PROMPT, INITIATIVE_CLASSIFICATION_PROMPT

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.environ.get("openai_api_key"),
    model_name="cognitivecomputations/dolphin-2.9-llama3-8b",
    openai_api_base="https://bbiopuow45en08-8000.proxy.runpod.net/v1",
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