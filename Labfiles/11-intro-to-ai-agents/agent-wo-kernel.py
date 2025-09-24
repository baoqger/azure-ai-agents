import os 
import asyncio
from typing import Annotated
from openai import AsyncOpenAI

from dotenv import load_dotenv

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior, PromptExecutionSettings
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.kernel import Kernel
import random   

# Define a sample plugin for the sample
class DestinationsPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        # List of vacation destinations
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia"
        ]
        # Track last destination to avoid repeats
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(self) -> Annotated[str, "Returns a random vacation destination."]:
        # Get available destinations (excluding last one if possible)
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)

        # Select a random destination
        destination = random.choice(available_destinations)

        # Update the last destination
        self.last_destination = destination

        return destination

load_dotenv()

api_key = os.environ.get("API_KEY")
deployment_name = os.environ.get("MODEL_DEPLOYMENT_NAME")
endpoint = os.environ.get("PROJECT_ENDPOINT")

# By providing the chat completion service directly

service_id = "agent"

ai_service = AzureChatCompletion(
    service_id=service_id, 
    api_key= api_key,
    deployment_name= deployment_name,
    endpoint=endpoint
)

# Configure the function choice behavior to auto invoke kernel functions
settings =  PromptExecutionSettings()
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

agent = ChatCompletionAgent(
    service=ai_service,
    name="TravelAgent",
    plugins=[DestinationsPlugin()],
    instructions="You are a helpful AI Agent that can help plan vacations for customers at random destinations",
    arguments=KernelArguments(settings=settings),
)

async def main():
    # Create a new thread for the agent
    # If no thread is provided, a new thread will be
    # created and returned with the initial response
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "Plan me a day trip.",
    ]

    for user_input in user_inputs:
        print(f"# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(
            messages=user_input, thread=thread,
        ):
            # 5. Print the response
            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(f"{response}", end="", flush=True)
            thread = response.thread
        print()

    # Clean up the thread
    await thread.delete() if thread else None

if __name__ == "__main__":
    asyncio.run(main()) 