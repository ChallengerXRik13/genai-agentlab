from backend.agents import run_rag_pipeline

def query_agent_lab(file, query):
    return run_rag_pipeline(file, query)
