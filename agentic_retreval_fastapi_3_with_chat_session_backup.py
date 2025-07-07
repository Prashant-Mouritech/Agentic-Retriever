import os
import time
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI
import logging
from datetime import datetime
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Environment variables
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
azure_openai_embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
azure_openai_embedding_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
azure_openai_chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")
SEARCH_INDEX_NAME = "your_search_index_name"  # Replace with your actual index name
EMBEDDING_VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", 1536))

# Validate environment variables
required_env_vars = {
    "AZURE_SEARCH_ENDPOINT": azure_search_endpoint,
    "AZURE_SEARCH_API_KEY": azure_search_key,
    "AZURE_OPENAI_KEY": azure_openai_key,
    "AZURE_OPENAI_EMBEDDING_ENDPOINT": azure_openai_embedding_endpoint
}
missing_vars = [var for var, value in required_env_vars.items() if value is None]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Retrieval API",
    description="API for agentic retrieval and chat response generation using Azure AI Search and Azure OpenAI.",
    version="1.0.0"
)

# Pydantic models for request and response
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_question: str
    source_filter: Optional[str] = None
    conversation_history: Optional[List[Message]] = None
    erase_conversation_history: bool = False

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict]
    conversation_history: List[Message]

# In-memory session store (session_id -> conversation_history)
sessions: Dict[str, List[Message]] = {}
current_session_id: str = "session_1"
session_counter: int = 1

# Dependency to initialize clients
async def get_clients():
    search_client = SearchClient(
        endpoint=azure_search_endpoint,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(azure_search_key)
    )
    embedding_client = AsyncAzureOpenAI(
        azure_endpoint=azure_openai_embedding_endpoint,
        api_key=azure_openai_key,
        api_version=azure_openai_version
    )
    chat_client = AsyncAzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        api_version=azure_openai_version
    )
    return search_client, embedding_client, chat_client

# Function to save conversation history to a file
def save_conversation_history(session_id: str, history: List[Message]):
    if not history:
        return
    # Get current timestamp in IST
    ist = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist).strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_history_{session_id}_{timestamp}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Timestamp: {timestamp} IST\n")
            f.write("Conversation History:\n")
            for msg in history:
                f.write(f"{msg.role}: {msg.content}\n")
        logger.info(f"Saved conversation history to {filename}")
    except Exception as e:
        logger.error(f"Error saving conversation history to file: {e}")

# Function to start a new session
def start_new_session() -> str:
    global current_session_id, session_counter
    session_counter += 1
    current_session_id = f"session_{session_counter}"
    sessions[current_session_id] = []
    # Log new session start with timestamp
    ist = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"New session started: {current_session_id} at {timestamp} IST")
    return current_session_id

async def generate_subqueries(query: str, conversation_history: Optional[List[Message]], chat_client: AsyncAzureOpenAI) -> List[str]:
    """
    Use an LLM to break down the query into subqueries for agentic retrieval.
    """
    try:
        history_text = ""
        if conversation_history:
            history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation_history])
            logger.info(f"Conversation History:\n{history_text}")
        else:
            logger.info("Conversation History: None provided")
            history_text = "No conversation history provided."

        prompt = (
            "You are an assistant that breaks down complex user queries into simpler subqueries for better information retrieval. "
            "Given the user query and conversation history, generate a list of focused subqueries that can be executed in parallel to retrieve relevant information. "
            "Return the subqueries as a list in a single line, separated by ' | '.\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"User query: {query}\n"
            "Subqueries:"
        )

        response = await chat_client.chat.completions.create(
            model=azure_openai_chat_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            temperature=0.5
        )

        subqueries_text = response.choices[0].message.content.strip()
        subqueries = subqueries_text.split(" | ")
        logger.info(f"Generated subqueries: {subqueries}")
        return subqueries if subqueries else [query]

    except Exception as e:
        logger.error(f"Error generating subqueries: {e}")
        return [query]

async def hybrid_search(
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
    conversation_history: Optional[List[Message]] = None,
    max_docs_for_reranker: int = 50,
    search_client: SearchClient = None,
    embedding_client: AsyncAzureOpenAI = None,
    chat_client: AsyncAzureOpenAI = None
) -> Dict:
    """
    Perform agentic retrieval using Azure AI Search.
    """
    try:
        # Step 1: Query Planning - Break down the query into subqueries using LLM
        subqueries = await generate_subqueries(query, conversation_history, chat_client)

        # Step 2: Generate embeddings for the original query (for vector search)
        embedding_response = await embedding_client.embeddings.create(
            input=query,
            model=azure_openai_embedding_model_name
        )
        query_embedding = embedding_response.data[0].embedding

        # Validate embedding vector dimension
        if len(query_embedding) != EMBEDDING_VECTOR_DIM:
            raise ValueError(
                f"Embedding vector dimension mismatch: expected {EMBEDDING_VECTOR_DIM}, got {len(query_embedding)}"
            )

        # Prepare to collect results and activities
        all_results = []
        activities = []

        # Step 3: Parallel Subquery Execution
        for subquery in subqueries:
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=min(max_docs_for_reranker, top_k),
                fields="embedding"
            )

            search_params = {
                "search_text": subquery,
                "vector_queries": [vector_query],
                "top": min(max_docs_for_reranker, top_k),
                "select": [
                    "id", "file_name", "file_url", "chunk", "content",
                    "lastModifiedDateTime", "lastModifiedBy", "source", "metadata"
                ]
            }

            if source_filter:
                search_params["filter"] = f"source eq '{source_filter}'"

            start_time = time.time()
            results = search_client.search(**search_params)
            subquery_results = []
            for result in results:
                subquery_results.append({
                    "id": result["id"],
                    "file_name": result["file_name"],
                    "file_url": result["file_url"],
                    "chunk": result.get("chunk", "Not available"),
                    "content": result.get("content", "Not available"),
                    "lastModifiedDateTime": result["lastModifiedDateTime"],
                    "lastModifiedBy": result["lastModifiedBy"],
                    "source": result.get("source", "Not available"),
                    "metadata": result.get("metadata", "Not available"),
                    "score": result.get("@search.score", 0),
                    "reranker_score": result.get("@search.rerankerScore", 0)
                })

            execution_time = time.time() - start_time
            activities.append({
                "subquery": subquery,
                "hit_count": len(subquery_results),
                "execution_time_seconds": execution_time,
                "filters_applied": f"source eq '{source_filter}'" if source_filter else "None"
            })

            all_results.extend(subquery_results)

        # Step 4: Merge and Rerank Results
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result["file_url"] not in seen_urls:
                seen_urls.add(result["file_url"])
                unique_results.append(result)

        unique_results.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
        top_results = unique_results[:top_k]

        # Step 5: Construct the Three-Part Response
        unified_result = "\n".join([result["chunk"] for result in top_results])
        references = [
            {
                "file_name": result["file_name"],
                "file_url": result["file_url"],
                "lastModifiedDateTime": result["lastModifiedDateTime"],
                "lastModifiedBy": result["lastModifiedBy"],
                "source": result["source"]
            }
            for result in top_results
        ]

        return {
            "unified_result": unified_result,
            "references": references,
            "activities": activities
        }

    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}")
        return {
            "unified_result": "",
            "references": [],
            "activities": [{"error": str(e)}]
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    clients: tuple = Depends(get_clients)
):
    """
    Generate a chat response using agentic retrieval results.
    """
    global current_session_id

    user_question = request.user_question
    source_filter = request.source_filter
    conversation_history = request.conversation_history
    erase_conversation_history = request.erase_conversation_history

    # Log the raw conversation history for debugging
    logger.info(f"Raw conversation history received: {conversation_history}")

    # Validate user_question
    if not isinstance(user_question, str):
        logger.error(f"Error: User question must be a string, got type {type(user_question)}: {user_question}")
        raise HTTPException(status_code=400, detail={
            "answer": "Invalid question provided. Please enter a valid question.",
            "citations": [],
            "conversation_history": conversation_history or []
        })

    user_question = user_question.strip()
    if not user_question:
        logger.error("Error: User question is empty after stripping")
        raise HTTPException(status_code=400, detail={
            "answer": "Invalid question provided. Please enter a valid question.",
            "citations": [],
            "conversation_history": conversation_history or []
        })

    try:
        # Handle session and conversation history
        if erase_conversation_history:
            # Save current conversation history to file before erasing
            if current_session_id in sessions:
                save_conversation_history(current_session_id, sessions[current_session_id])
            # Start a new session
            current_session_id = start_new_session()
            # Ignore provided conversation_history since we're starting fresh
            conversation_history = None

        # Determine if this is the first chat in the session
        is_first_chat = current_session_id not in sessions or len(sessions.get(current_session_id, [])) == 0

        # For the first chat, ignore any provided conversation_history and start fresh
        if is_first_chat:
            logger.info("This is the first chat in the session. Ignoring provided conversation_history.")
            conversation_history = []
            sessions[current_session_id] = []  # Ensure session history is empty
        else:
            # For subsequent chats, always use the session's conversation_history
            logger.info("This is a subsequent chat. Using session's conversation_history.")
            conversation_history = sessions.get(current_session_id, [])

        logger.info(f"Processing user question: '{user_question}' with source filter: '{source_filter}'")
        search_client, embedding_client, chat_client = clients

        # Step 1: Perform agentic retrieval using hybrid_search
        search_response = await hybrid_search(
            query=user_question,
            top_k=5,
            source_filter=source_filter,
            conversation_history=conversation_history,
            max_docs_for_reranker=50,
            search_client=search_client,
            embedding_client=embedding_client,
            chat_client=chat_client
        )

        unified_result = search_response["unified_result"]
        citations = search_response["references"]
        activities = search_response["activities"]

        # Log the search activities for debugging/monitoring
        logger.info(f"Search activities: {activities}")

        if not unified_result:
            updated_history = conversation_history.copy() if conversation_history else []
            updated_history.append(Message(role="user", content=user_question))
            updated_history.append(Message(role="assistant", content="No relevant information found to answer your question."))
            # Store updated history in session
            sessions[current_session_id] = updated_history
            return ChatResponse(
                answer="No relevant information found to answer your question.",
                citations=citations,
                conversation_history=updated_history
            )

        # Step 2: Construct the prompt for final answer generation
        prompt = (
            "You are an assistant providing accurate answers based on the following context from company documents. "
            "Use only the provided information to answer the question. For questions about a process or procedure, extract and include all relevant steps, details, or guidelines from the provided context. "
            "If the context lacks specific details, provide what is available and note any limitations. Do not generate answers beyond the context.\n\n"
            "Context:\n"
            f"{unified_result}\n\n"
            "Activity Log (for reference):\n"
            f"{activities}\n\n"
            f"User question: {user_question}\n"
            "Answer:"
        )

        # Step 3: Generate the final answer using Azure OpenAI
        try:
            client = AsyncAzureOpenAI(
                azure_deployment=azure_openai_chat_model,
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_key,
                api_version=azure_openai_version
            )
            response = await client.chat.completions.create(
                model=azure_openai_chat_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=500,
                temperature=0.7
            )
            output = response.choices[0].message.content.strip()

            updated_history = conversation_history.copy() if conversation_history else []
            updated_history.append(Message(role="user", content=user_question))
            updated_history.append(Message(role="assistant", content=output))
            # Store updated history in session
            sessions[current_session_id] = updated_history

            return ChatResponse(
                answer=output,
                citations=citations,
                conversation_history=updated_history
            )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            updated_history = conversation_history.copy() if conversation_history else []
            updated_history.append(Message(role="user", content=user_question))
            updated_history.append(Message(role="assistant", content="Unable to generate a response due to an error."))
            # Store updated history in session
            sessions[current_session_id] = updated_history
            return ChatResponse(
                answer="Unable to generate a response due to an error.",
                citations=citations,
                conversation_history=updated_history
            )

    except Exception as e:
        logger.error(f"Error in get_chat_response: {e}")
        updated_history = conversation_history.copy() if conversation_history else []
        updated_history.append(Message(role="user", content=user_question))
        updated_history.append(Message(role="assistant", content="An error occurred while processing your question."))
        # Store updated history in session
        sessions[current_session_id] = updated_history
        return ChatResponse(
            answer="An error occurred while processing your question.",
            citations=[],
            conversation_history=updated_history
        )

if __name__ == "__main__":
    import uvicorn
    # Log initial session start
    start_new_session()
    uvicorn.run(app, host="0.0.0.0", port=8000)