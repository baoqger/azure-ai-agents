import asyncio
import os
import sys
from datetime import datetime

from azure.identity import AzureCliCredential

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel

async def main():
    kernel = Kernel()

    # Add the AzureChatCompletion AI Service to the Kernel
    service_id = "agent"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id, 
            api_key="FKfCvE2h93cUfV4tL60M7hYelCyApkVS4NrRUJt0GL99SlDLBSKLJQQJ99BHACHYHv6XJ3w3AAAAACOGayCs",
            deployment_name="gpt-4o",
            endpoint="https://jbao6-8161-resource.openai.azure.com/"
        )
    )

    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    # Configure the function choice behavior to auto invoke kernel functions
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Set your GitHub Personal Access Token (PAT) value here
    # gh_settings = GitHubSettings(token="")  # nosec
    # kernel.add_plugin(plugin=GitHubPlugin(gh_settings), plugin_name="GithubPlugin")

    current_time = datetime.now().isoformat()

    # Create the agent
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="SampleAssistantAgent",
        instructions=f"""
            You are an agent designed to query and retrieve information from a single GitHub repository in a read-only 
            manner.
            You are also able to access the profile of the active user.

            Use the current date and time to provide up-to-date details or time-sensitive responses.
            
            The repository you are querying is a public repository with the following name: microsoft/semantic-kernel

            The current date and time is: {current_time}. 
            """,
        arguments=KernelArguments(settings=settings),
    )

    thread: ChatHistoryAgentThread = None
    is_complete: bool = False
    while not is_complete:
        user_input = input("User:> ")
        if not user_input:
            continue

        if user_input.lower() == "exit":
            is_complete = True
            break

        arguments = KernelArguments(now=datetime.now().strftime("%Y-%m-%d %H:%M"))

        async for response in agent.invoke(messages=user_input, thread=thread, arguments=arguments):
            print(f"{response.content}")
            thread = response.thread


if __name__ == "__main__":
    asyncio.run(main())