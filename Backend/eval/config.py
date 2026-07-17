import os 
from langchain_groq import ChatGroq
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

JUDGE_LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def get_judge_llm() :
    raw_llm = ChatGroq(
        model= JUDGE_LLM_MODEL,
        temperature=0,
        groq_api_key = GROQ_API_KEY,
    )
    return LangchainLLMWrapper(raw_llm)

def get_judge_embeddings():
    raw_embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    return LangchainEmbeddingsWrapper(raw_embeddings)


EVAL_RUN_CONFIG =RunConfig(max_workers=3,timeout=180)