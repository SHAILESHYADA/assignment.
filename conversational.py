import asyncio
from agents.base_agent import BaseAgent
from agents.handle_agent_errors import handle_agent_error
from config import Config
from memory.zep.zep_memory import ZepMemory
from services.pubsub import ChatPubSubService
from services.run_log import RunLogsManager
from services.voice import speech_to_text, text_to_speech
from typings.agent import AgentWithConfigsOutput
from typings.config import AccountSettings, AccountVoiceSettings
from utils.system_message import SystemMessageBuilder
from xagent.core import XAgent

class ConversationalAgent(BaseAgent):
    async def run(
        self,
        settings: AccountSettings,
        voice_settings: AccountVoiceSettings,
        chat_pubsub_service: ChatPubSubService,
        agent_with_configs: AgentWithConfigsOutput,
        tools,
        prompt: str,
        voice_url: str,
        history: PostgresChatMessageHistory,
        human_message_id: str,
        run_logs_manager: RunLogsManager,
        pre_retrieved_context: str,
    ):
        memory = ZepMemory(
            session_id=str(self.session_id),
            url=Config.ZEP_API_URL,
            api_key=Config.ZEP_API_KEY,
            memory_key="chat_history",
            return_messages=True,
        )

        memory.human_name = self.sender_name
        memory.ai_name = agent_with_configs.agent.name

        system_message = SystemMessageBuilder(
            agent_with_configs, pre_retrieved_context
        ).build()

        try:
            if voice_url:
                configs = agent_with_configs.configs
                prompt = speech_to_text(voice_url, configs, voice_settings)

            # Initialize the XAgent
            agent = XAgent(
                tools=tools,
                memory=memory,
                system_prompt=system_message,
                verbose=True,
            )

            # Interact with XAgent
            response_chunks = []
            async for response in agent.stream_response({"input": prompt}):
                response_chunks.append(response)
                yield response

            # Combine all response chunks for final output
            final_response = "".join(response_chunks)

        except Exception as err:
            final_response = handle_agent_error(err)
            memory.save_context(
                {"input": prompt, "chat_history": memory.load_memory_variables({})["chat_history"]},
                {"output": final_response},
            )
            yield final_response

        try:
            configs = agent_with_configs.configs
            voice_url = None
            if "Voice" in configs.response_mode:
                voice_url = text_to_speech(final_response, configs, voice_settings)
        except Exception as err:
            final_response = f"{final_response}\n\n{handle_agent_error(err)}"
            yield final_response

        ai_message = history.create_ai_message(
            final_response,
            human_message_id,
            agent_with_configs.agent.id,
            voice_url,
        )

        chat_pubsub_service.send_chat_message(chat_message=ai_message)
