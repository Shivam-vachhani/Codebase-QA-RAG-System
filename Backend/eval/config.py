from eval import _ragas_compat
import os 
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from dotenv import load_dotenv


load_dotenv()


JUDGE_LLM_MODEL = "gpt-4.1-nano"

def get_judge_llm() :
    raw_llm = ChatOpenAI(model= JUDGE_LLM_MODEL,temperature=0)
    return LangchainLLMWrapper(raw_llm)

def get_judge_embeddings():
    raw_embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    return LangchainEmbeddingsWrapper(raw_embeddings)


EVAL_RUN_CONFIG =RunConfig(max_workers=4,timeout=180)