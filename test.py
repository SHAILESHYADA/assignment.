from xagent.core import XAgent
from xagent.tools.search import GoogleSearchTool
from xagent.evaluation import EvalConfig, run_eval_on_dataset
from langsmith import Client
from config import Config

def agent_factory():
    """
    Creates and returns an XAgent instance configured with tools.
    """
    # Initialize tools (e.g., a Google search tool)
    tools = [GoogleSearchTool(api_key=Config.GOOGLE_API_KEY)]

    # Initialize XAgent with the tools and system message
    agent = XAgent(
        tools=tools,
        system_prompt="You are a helpful assistant skilled in answering questions concisely and helpfully.",
        verbose=True,
        max_iterations=5,
    )
    return agent


# Create the agent
agent = agent_factory()

# Initialize LangSmith client
client = Client()

# Define evaluation configuration
eval_config = EvalConfig(
    evaluators=[
        "qa",
        EvalConfig.Criteria("helpfulness"),
        EvalConfig.Criteria("conciseness"),
    ],
    input_key="input",
    evaluation_agent=agent,  # Use the XAgent for evaluation
)

# Run the evaluation on a dataset
chain_results = run_eval_on_dataset(
    client=client,
    dataset_name="test-dataset",
    agent=agent,  # Use XAgent for dataset processing
    evaluation_config=eval_config,
    concurrency_level=1,
    verbose=True,
)
