import json
from pathlib import Path
import uuid
from typing import Any, Callable, Set

# Create a function can be created to provide a greeting message to users. 
def greeding(username: str) -> str:
     """
     Provide a greeting message to users.

     :param username: the name of the user.
     :return: greeting message.
     """ 
     return f"Hello {username}, welcome! "

def get_weather(city: str) -> str:
     """
     Get the weather information for your city.

     :param city: the name of the city.
     :return: weather information as a JSON object.
     """ 
     data = {
        "city": city,
        "temperature": 30
     }
     return data

# Create a function to submit a support ticket
def submit_support_ticket(email_address: str, description: str) -> str:
     """
     Submit a support ticket.

     :param email_address: email address of the user.
     :param description: description of the issue.
     :return: ticket information as a JSON string.
     """     
     script_dir = Path(__file__).parent  # Get the directory of the script
     ticket_number = str(uuid.uuid4()).replace('-', '')[:6]
     file_name = f"ticket-{ticket_number}.txt"
     file_path = script_dir / file_name
     text = f"Support ticket: {ticket_number}\nSubmitted by: {email_address}\nDescription:\n{description}"
     file_path.write_text(text)
    
     message_json = json.dumps({"message": f"Support ticket {ticket_number} submitted. The ticket file is saved as {file_name}"})
     return message_json

# Define a set of callable functions
user_functions: Set[Callable[..., Any]] = {
     get_weather,
     greeding
 }

