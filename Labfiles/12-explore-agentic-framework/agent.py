import json
import os 
import asyncio

from typing import Annotated

from dotenv import load_dotenv

# from IPython.display import display, HTML

from openai import AsyncOpenAI

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior, PromptExecutionSettings
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent, ChatMessageContent
from semantic_kernel.functions import kernel_function, KernelArguments

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
    plugins=[DestinationsPlugin()],
    name="TravelAgent",
    instructions="You are a helpful AI Agent that can help plan vacations for customers at random destinations",
    arguments=KernelArguments(settings=settings),
)

user_inputs = [
    "Plan me a day trip.",
    "I don't like that destination. Plan me another vacation.",
]

# This callback function will be called for each intermediate message
# Which will allow one to handle FunctionCallContent and FunctionResultContent
# If the callback is not provided, the agent will return the final response
# with no intermediate tool call steps.
async def handle_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        if isinstance(item, FunctionCallContent):
            print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        elif isinstance(item, FunctionResultContent):
            print(f"Function Result:> {item.result} for function: {item.name}")
        else:
            print(f"{message.role}: {message.content}")

async def main():
    thread: ChatHistoryAgentThread | None = None

    for user_input in user_inputs:
        print(f"# User: {user_input}\n")

        async for response in agent.invoke(
            messages=user_input,
            thread=thread,
            on_intermediate_message=handle_intermediate_steps,
        ):
            print(f"# {response.role}: {response}")
            thread = response.thread

if __name__ == "__main__":
    asyncio.run(main()) 