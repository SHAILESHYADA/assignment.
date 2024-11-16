from typing import List, Optional

from xagent.core import XAgent
from langchain.schema import AIMessage, SystemMessage

from agents.agent_simulations.agent.dialogue_agent import DialogueAgent
from config import Config
from memory.zep.zep_memory import ZepMemory
from services.run_log import RunLogsManager
from typings.agent import AgentWithConfigsOutput


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        agent_with_configs: AgentWithConfigsOutput,
        system_message: SystemMessage,
        model: any,  # Replace with XAgent-compatible model type if available
        tools: List[any],
        session_id: str,
        sender_name: str,
        is_memory: bool = False,
        run_logs_manager: Optional[RunLogsManager] = None,
        **tool_kwargs,
    ) -> None:
        super().__init__(name, agent_with_configs, system_message, model)
        self.tools = tools
        self.session_id = session_id
        self.sender_name = sender_name
        self.is_memory = is_memory
        self.run_logs_manager = run_logs_manager

    def send(self) -> str:
        """
        Applies the XAgent to the message history
        and returns the message string
        """

        # Initialize memory using ZepMemory
        memory = ZepMemory(
            session_id=self.session_id,
            url=Config.ZEP_API_URL,
            api_key=Config.ZEP_API_KEY,
            memory_key="chat_history",
            return_messages=True,
        )

        memory.human_name = self.sender_name
        memory.ai_name = self.agent_with_configs.agent.name
        memory.auto_save = False

        # Initialize XAgent
        agent = XAgent(
            tools=self.tools,
            memory=memory,
            system_prompt=self.system_message.content,
            verbose=True,
        )

        # Combine message history and prefix into a single prompt
        prompt = "\n".join(self.message_history + [self.prefix])

        # Get agent's response
        try:
            response_chunks = []
            for chunk in agent.get_response({"input": prompt}):
                response_chunks.append(chunk)

            res = "".join(response_chunks)
        except Exception as err:
            res = f"Error during agent execution: {str(err)}"

        # Save the AI's response to memory if needed
        # memory.save_ai_message(res)

        # Wrap the result in an AIMessage and return its content
        message = AIMessage(content=res)
        return message.content
