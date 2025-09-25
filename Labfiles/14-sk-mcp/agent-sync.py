import os 
import asyncio
from typing import Annotated
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin, MCPStdioPlugin
from semantic_kernel.connectors.ai import FunctionChoiceBehavior, PromptExecutionSettings
from semantic_kernel.functions import kernel_function, KernelArguments

load_dotenv()

service_id = "agent"
instrutions = """
        You are an inventory assistant. Here are some general guidelines:
        - Recommend restock if item inventory < 10  and weekly sales > 15
        - Recommend clearance if item inventory > 20 and weekly sales < 5
        """
api_key = os.environ.get("API_KEY")
deployment_name = os.environ.get("MODEL_DEPLOYMENT_NAME")
endpoint = os.environ.get("PROJECT_ENDPOINT")

async def main():

        plugin = MCPStdioPlugin(
            name="inventory",
            description="inventory Plugin",
            command="python",
            args=[".\\src\\local-mcp\\server.py"],
            env={},
        )

        await plugin.connect()

        # Configure the function choice behavior to auto invoke kernel functions
        settings =  PromptExecutionSettings()
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        # Now create our agent and plug in the MCP plugin
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(
                service_id=service_id, 
                api_key= api_key,
                deployment_name= deployment_name,
                endpoint=endpoint
            ),
            name="ChatBot",
            instructions=instrutions,
            plugins=[plugin],
            arguments=KernelArguments(settings=settings),
        )
        print("Agent initialized")
        print(f"agent: {agent}")
        thread: ChatHistoryAgentThread = None
        user_messages = [
            "What are the current inventory levels?",
            "Are there any products that should be restocked?",
            ]

        for user_message in user_messages:
            print("*** User:", user_message)
            response = await agent.get_response(messages=user_message, thread=thread)
            thread = response.thread
            print("*** Agent:", response.content)

        await plugin.close()

if __name__ == "__main__":
    asyncio.run(main())