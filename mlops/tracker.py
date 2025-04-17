import wandb

def init_tracking():
    wandb.init(project="genai-agentlab", name="session")

def log_interaction(query, answer):
    wandb.log({"question": query, "response": answer})
