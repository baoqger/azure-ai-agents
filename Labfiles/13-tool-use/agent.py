import requests
import json
import os 
import asyncio
from typing import Annotated
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior, PromptExecutionSettings
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent, ChatMessageContent
from semantic_kernel.functions import kernel_function, KernelArguments

# Define Booking Plugin
class BookingPlugin:
    """Booking Plugin for customers"""

    @kernel_function(description="booking hotel")
    def booking_hotel(
        self, 
        query: Annotated[str, "The name of the city"], 
        check_in_date: Annotated[str, "Hotel Check-in Time"], 
        check_out_date: Annotated[str, "Hotel Check-out Time"],
    ) -> Annotated[str, "Return the result of booking hotel information"]:
        """
        Function to book a hotel.
        Parameters:
        - query: The name of the city
        - check_in_date: Hotel Check-in Time
        - check_out_date: Hotel Check-out Time
        Returns:
        - The result of booking hotel information
        """

        # Define the parameters for the hotel booking request
        params = {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": "1",
            "currency": "GBP",
            "gl": "uk",
            "hl": "en",
            "api_key": SERP_API_KEY
        }

        # Send the GET request to the SERP API
        response = requests.get(SERP_BASE_URL, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response content as JSON
            response = response.json()
            # Return the properties from the response
            return response["properties"]
        else:
            # Return None if the request failed
            return None

    @kernel_function(description="booking flight")
    def booking_flight(
        self, 
        origin: Annotated[str, "The name of Departure"], 
        destination: Annotated[str, "The name of Destination"], 
        outbound_date: Annotated[str, "The date of outbound"], 
        return_date: Annotated[str, "The date of Return_date"],
    ) -> Annotated[str, "Return the result of booking flight information"]:
        """
        Function to book a flight.
        Parameters:
        - origin: The name of Departure
        - destination: The name of Destination
        - outbound_date: The date of outbound
        - return_date: The date of Return_date
        - airline: The preferred airline carrier
        - hotel_brand: The preferred hotel brand
        Returns:
        - The result of booking flight information
        """
        
        # Define the parameters for the outbound flight request
        go_params = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "currency": "GBP",
            "hl": "en",
            "airline": "airline",
            "hotel_brand": "hotel_brand",
            "api_key": SERP_API_KEY
        }
 
        print(go_params)

        # Send the GET request for the outbound flight
        go_response = requests.get(SERP_BASE_URL, params=go_params)

        # Initialize the result string
        result = ''

        # Check if the outbound flight request was successful
        if go_response.status_code == 200:
            # Parse the response content as JSON
            response = go_response.json()
            # Append the outbound flight information to the result
            result += "# outbound \n " + str(response)
        else:
            # Print an error message if the request failed
            print('error!!!')

        # Define the parameters for the return flight request
        back_params = {
            "engine": "google_flights",
            "departure_id": destination,
            "arrival_id": origin,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "currency": "GBP",
            "hl": "en",
            "api_key": SERP_API_KEY
        }

        # Send the GET request for the return flight
        back_response = requests.get(SERP_BASE_URL, params=back_params)

        # Check if the return flight request was successful
        if back_response.status_code == 200:
            # Parse the response content as JSON
            response = back_response.json()
            # Append the return flight information to the result
            result += "\n # return \n" + str(response)
        else:
            # Print an error message if the request failed
            print('error!!!')

        # Print the result
        print(result)

        # Return the result
        return result
    

load_dotenv()

api_key = os.environ.get("API_KEY")
deployment_name = os.environ.get("MODEL_DEPLOYMENT_NAME")
endpoint = os.environ.get("PROJECT_ENDPOINT")
SERP_API_KEY = os.environ.get("SERP_API_KEY")
SERP_BASE_URL = os.environ.get("SERP_BASE_URL")


# By providing the chat completion service directly

# Define the agent's name and instructions
service_id = "BookingAgent"
AGENT_INSTRUCTIONS = """
You are a booking agent, help me to book flights or hotels.

Thought: Understand the user's intention and confirm whether to use the reservation system to complete the task.

Action:
- If booking a flight, convert the departure name and destination name into airport codes.
  An airport code is an uppercase 3-letter code.
  For example, CDG is Paris Charles de Gaulle Airport and AUS is Austin-Bergstrom International Airport.
- If booking a hotel or flight, use the corresponding API to call. Ensure that the necessary parameters are available. If any parameters are missing, use default values or assumptions to proceed.
- If it is not a hotel or flight booking, respond with the final answer only.
- Output the results using a markdown table:
- For flight bookings, separate the outbound and return contents and list them in the order of Departure_airport Name | Airline | Flight Number | Departure Time | Arrival_airport Name | Arrival Time | Duration | Airplane | Travel Class | Price (USD) | Legroom | Extensions | Carbon Emissions (kg).
- For hotel bookings, list them in the order of Properties Name | Properties description | check_in_time | check_out_time | prices | nearby_places | hotel_class | gps_coordinates.
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
    plugins=[BookingPlugin()],
    name="BookingAgent",
    instructions=AGENT_INSTRUCTIONS,
    arguments=KernelArguments(settings=settings),
)

# This is your prompt for the activity or task you want to complete 
# Define user inputs for the agent to process we have provided some example prompts to test and validate 
user_inputs = [
    # "Can you tell me the round-trip air ticket from  London to New York JFK aiport, the departure time is February 17, 2025, and the return time is February 23, 2025"
    # "Book a hotel in New York from Feb 20,2025 to Feb 24,2025"
    "Help me book flight tickets and hotel for the following trip London Heathrow LHR Oct 20th 2025 to New York JFK returning Oct 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York please provide costs for the flight and hotel"
    # "I have a business trip from London LHR to New York JFK on Feb 20th 2025 to Feb 27th 2025, can you help me to book a hotel and flight tickets"
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