import requests
import json
import os 
import asyncio
from chromadb import chromadb, Collection
from typing import Annotated
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior, PromptExecutionSettings
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent, ChatMessageContent
from semantic_kernel.functions import kernel_function, KernelArguments


class WeatherInfoPlugin:
    """A Plugin that provides the average temperature for a travel destination."""

    def __init__(self):
        # Dictionary of destinations and their average temperatures
        self.destination_temperatures = {
            "maldives": "82°F (28°C)",
            "swiss alps": "45°F (7°C)",
            "african safaris": "75°F (24°C)"
        }

    @kernel_function(description="Get the average temperature for a specific travel destination.")
    def get_destination_temperature(self, destination: str) -> Annotated[str, "Returns the average temperature for the destination."]:
        """Get the average temperature for a travel destination."""
        # Normalize the input destination (lowercase)
        normalized_destination = destination.lower()

        # Look up the temperature for the destination
        if normalized_destination in self.destination_temperatures:
            return f"The average temperature in {destination} is {self.destination_temperatures[normalized_destination]}."
        else:
            return f"Sorry, I don't have temperature information for {destination}. Available destinations are: Maldives, Swiss Alps, and African safaris."

class DestinationsPlugin:
    # Destination data store with rich details about popular travel locations
    DESTINATIONS = {
        "maldives": {
            "name": "The Maldives",
            "description": "An archipelago of 26 atolls in the Indian Ocean, known for pristine beaches and overwater bungalows.",
            "best_time": "November to April (dry season)",
            "activities": ["Snorkeling", "Diving", "Island hopping", "Spa retreats", "Underwater dining"],
            "avg_cost": "$400-1200 per night for luxury resorts"
        },
        "swiss alps": {
            "name": "The Swiss Alps",
            "description": "Mountain range spanning across Switzerland with picturesque villages and world-class ski resorts.",
            "best_time": "December to March for skiing, June to September for hiking",
            "activities": ["Skiing", "Snowboarding", "Hiking", "Mountain biking", "Paragliding"],
            "avg_cost": "$250-500 per night for alpine accommodations"
        },
        "safari": {
            "name": "African Safari",
            "description": "Wildlife viewing experiences across various African countries including Kenya, Tanzania, and South Africa.",
            "best_time": "June to October (dry season) for optimal wildlife viewing",
            "activities": ["Game drives", "Walking safaris", "Hot air balloon rides", "Cultural village visits"],
            "avg_cost": "$400-800 per person per day for luxury safari packages"
        },
        "bali": {
            "name": "Bali, Indonesia",
            "description": "Island paradise known for lush rice terraces, beautiful temples, and vibrant culture.",
            "best_time": "April to October (dry season)",
            "activities": ["Surfing", "Temple visits", "Rice terrace trekking", "Yoga retreats", "Beach relaxation"],
            "avg_cost": "$100-500 per night depending on accommodation type"
        },
        "santorini": {
            "name": "Santorini, Greece",
            "description": "Stunning volcanic island with white-washed buildings and blue domes overlooking the Aegean Sea.",
            "best_time": "Late April to early November",
            "activities": ["Sunset watching in Oia", "Wine tasting", "Boat tours", "Beach hopping", "Ancient ruins exploration"],
            "avg_cost": "$200-600 per night for caldera view accommodations"
        }
    }

    @kernel_function(
        name="get_destination_info",
        description="Provides detailed information about specific travel destinations."
    )
    def get_destination_info(self, query: str) -> str:
        # Find which destination is being asked about
        query_lower = query.lower()
        matching_destinations = []

        for key, details in DestinationsPlugin.DESTINATIONS.items():
            if key in query_lower or details["name"].lower() in query_lower:
                matching_destinations.append(details)

        if not matching_destinations:
            return (f"User Query: {query}\n\n"
                    f"I couldn't find specific destination information in our database. "
                    f"Please use the general retrieval system for this query.")

        # Format destination information
        destination_info = "\n\n".join([
            f"Destination: {dest['name']}\n"
            f"Description: {dest['description']}\n"
            f"Best time to visit: {dest['best_time']}\n"
            f"Popular activities: {', '.join(dest['activities'])}\n"
            f"Average cost: {dest['avg_cost']}" for dest in matching_destinations
        ])

        return (f"Destination Information:\n{destination_info}\n\n"
                f"User Query: {query}\n\n"
                "Based on the above destination details, provide a helpful response "
                "that addresses the user's query about this location.")

class PromptPlugin:

    def __init__(self, collection: Collection):
        self.collection = collection

    @kernel_function(
        name="build_augmented_prompt",
        description="Build an augmented prompt using retrieval context."
    )
    def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
        return (
            f"Retrieved Context:\n{retrieval_context}\n\n"
            f"User Query: {query}\n\n"
            "Based ONLY on the above context, please provide your answer."
        )
    
    @kernel_function(name="retrieve_context", description="Retrieve context from the database.")
    def get_retrieval_context(self, query: str) -> str:
        results = self.collection.query(
            query_texts=[query],
            include=["documents", "metadatas"],
            n_results=2
        )
        context_entries = []
        if results and results.get("documents") and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                context_entries.append(f"Document: {doc}\nMetadata: {meta}")
        return "\n\n".join(context_entries) if context_entries else "No retrieval context found."

# Initialize ChromaDB and create a collection for travel documents
collection = chromadb.PersistentClient(path="./chroma_db").create_collection(
    name="travel_documents",
    metadata={"description": "travel_service"},
    get_or_create=True,
)

documents = [
    "Contoso Travel offers hot destination: Maldives",
    "Contoso Travel offers luxury vacation packages to exotic destinations worldwide.",
    "Our premium travel services include personalized itinerary planning and 24/7 concierge support.",
    "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage.",
    "Popular destinations include the Maldives, Swiss Alps, and African safaris.",
    "Contoso Travel provides exclusive access to boutique hotels and private guided tours.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": "training", "type": "explanation"} for _ in documents]
)

load_dotenv()

api_key = os.environ.get("API_KEY")
deployment_name = os.environ.get("MODEL_DEPLOYMENT_NAME")
endpoint = os.environ.get("PROJECT_ENDPOINT")

# Define the agent's name and instructions
service_id = "BookingAgent"
AGENT_INSTRUCTIONS = """
Answer travel queries using the provided context. If context is provided, do not say 'I have no context for that.'
"""

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
    plugins=[DestinationsPlugin(), WeatherInfoPlugin(), PromptPlugin(collection)],
    name="TravelAgent",
    instructions=AGENT_INSTRUCTIONS,
    arguments=KernelArguments(settings=settings),
)


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
    print("The following is Agent's Response: ")

user_inputs = [
    "What is a good hot destination offered by Contoso, what is it average temperature and can you provide some general information of it?",
]

async def main():
    thread: ChatHistoryAgentThread | None = None

    for user_input in user_inputs:
        print(f"# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(
            messages=user_input,
            thread=thread,
            on_intermediate_message=handle_intermediate_steps,
        ):
            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(f"{response}", end="", flush=True)
            thread = response.thread
        print("\n")

if __name__ == "__main__":
    asyncio.run(main()) 