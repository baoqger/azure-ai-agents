from fastapi import APIRouter, HTTPException
from typing import List
from ..models import TaskItem, TaskCreateRequest, TaskUpdateRequest, ChatRequest, ChatMessage
from ..agents import LangGraphTaskAgent, FoundryTaskAgent


def create_api_routes(
    langgraph_agent: LangGraphTaskAgent,
    foundry_agent: FoundryTaskAgent
) -> APIRouter:
    """
    Create API router with task CRUD endpoints and chat agent routes.
    
    Routes:
    - POST   /chat/langgraph : Processes a chat message using the LangGraph agent
    - POST   /chat/foundry   : Processes a chat message using the Foundry agent
    """
    router = APIRouter()
    
    @router.post("/chat/langgraph", response_model=ChatMessage, operation_id="chatWithLangGraph", include_in_schema=False)
    async def chat_with_langgraph(chat_request: ChatRequest):
        """Process a chat message using the LangGraph agent"""
        try:
            if not chat_request.message:
                raise HTTPException(status_code=400, detail="Message is required")
            
            response = await langgraph_agent.process_message(
                chat_request.message, 
                chat_request.sessionId
            )
            return response
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error in LangGraph chat: {e}")
            raise HTTPException(status_code=500, detail="Failed to process message")
    
    @router.post("/chat/foundry", response_model=ChatMessage, operation_id="chatWithFoundry", include_in_schema=False)
    async def chat_with_foundry(chat_request: ChatRequest):
        """Process a chat message using the Foundry agent"""
        try:
            if not chat_request.message:
                raise HTTPException(status_code=400, detail="Message is required")
            
            response = await foundry_agent.process_message(chat_request.message)
            return response
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error in Foundry chat: {e}")
            raise HTTPException(status_code=500, detail="Failed to process message")
    
    return router
