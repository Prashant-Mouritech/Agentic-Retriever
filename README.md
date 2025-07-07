Agentic Retrieval System
This project implements an Agentic Retrieval System using Azure AI Search and Azure OpenAI to deliver intelligent, context-aware responses for chat applications. The system leverages a hybrid search approach, combining vector search (using HNSW) and keyword search (using BM25), enhanced by an LLM-driven query breakdown for superior retrieval performance. It is built with FastAPI for a high-performance API and supports session-based conversation history.
The source code is available at https://github.com/Prashant-Mouritech/Agentic-Retriever.
Features

Agentic Retrieval: Dynamically breaks down complex queries into subqueries using an LLM, improving relevance for multi-part questions.
Hybrid Search: Combines vector search (HNSW algorithm) for semantic understanding and keyword search (BM25 algorithm) for exact matches.
Session Management: Maintains conversation history in-memory for context-aware responses.
FastAPI Integration: Provides a scalable, async API for handling chat requests.
Azure AI Search: Leverages enterprise-grade search capabilities for efficient document retrieval.
Azure OpenAI: Uses models like gpt-4o-mini for subquery generation and answer synthesis.
Logging: Tracks system activity with timestamps in IST (Asia/Kolkata) for debugging and monitoring.
Cost Efficiency: Built without Azure Foundry, using standard Azure services to minimize costs.

Prerequisites
To run this project, ensure the following are installed and configured:

Python 3.8+
Azure Account with access to:
Azure AI Search
Azure OpenAI (for embeddings and chat completions)


Dependencies:
fastapi
uvicorn
python-dotenv
azure-search-documents
azure-core
openai
pydantic
typing



Installation

Clone the Repository:
git clone https://github.com/Prashant-Mouritech/Agentic-Retriever.git
cd Agentic-Retriever


Create a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Create a requirements.txt file with the following content:
fastapi==0.103.0
uvicorn==0.23.2
python-dotenv==1.0.0
azure-search-documents==11.4.0
azure-core==1.29.4
openai==1.10.0
pydantic==2.4.2


Set Up Environment Variables:Create a .env file in the project root with the following variables:
AZURE_SEARCH_ENDPOINT="your_azure_search_endpoint"
AZURE_SEARCH_API_KEY="your_azure_search_api_key"
AZURE_OPENAI_KEY="your_azure_openai_api_key"
AZURE_OPENAI_EMBEDDING_ENDPOINT="your_azure_openai_embedding_endpoint"
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
AZURE_OPENAI_EMBEDDING_MODEL_NAME="text-embedding-ada-002"
AZURE_OPENAI_CHAT_MODEL="gpt-4o-mini"
EMBEDDING_VECTOR_DIM=1536

Replace placeholders (e.g., your_azure_search_endpoint) with actual values from your Azure services.

Configure Azure AI Search Index:

Ensure an index named genai-book-index exists in your Azure AI Search instance.
The index should include fields like id, file_name, file_url, chunk, content, lastModifiedDateTime, lastModifiedBy, source, metadata, and embedding (for vector search).



Usage

Run the Application:
python main.py

This starts the FastAPI server on http://0.0.0.0:8000.

Access the API:

Endpoint: POST /chat
Request Body (JSON):{
  "user_question": "Your question here",
  "source_filter": null,
  "conversation_history": null,
  "erase_conversation_history": false
}


Example Request:curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"user_question": "How do I set up a project?", "source_filter": null, "conversation_history": null, "erase_conversation_history": false}'


Response:{
  "answer": "Response based on retrieved documents",
  "citations": [
    {
      "file_name": "doc1.pdf",
      "file





//   Example Response:    സ
   Example Response:
{
  "answer": "To set up a project, follow these steps: ...",
  "citations": [
    {
      "file_name": "project_management.pdf",
      "file_url": "https://example.com/doc1.pdf",
      "lastModifiedDateTime": "2023-10-01T12:00:00Z",
      "lastModifiedBy": "Admin",
      "source": "Internal"
    }
  ],
  "conversation_history": [
    {
      "role": "user",
      "content": "How do I set up a project?"
    },
    {
      "role": "assistant",
      "content": "To set up a project, follow these steps: ..."
    }
  ]
}


Features in Action:
The system breaks down complex queries into subqueries using the LLM.
Hybrid search combines HNSW (vector search) and BM25 (keyword search) for optimal results.
Conversation history is stored in-memory and saved to a file when a session is reset.



How It Works

Query Processing:

The generate_subqueries function uses Azure OpenAI to split complex queries into focused subqueries.
Example: "How do I set up a project and manage its timeline?" becomes ["project setup", "timeline management"].


Hybrid Search:

Vector Search: Uses HNSW algorithm for semantic similarity, powered by embeddings from text-embedding-ada-002.
Keyword Search: Uses BM25 algorithm for exact term matching.
Result Fusion: Reciprocal Rank Fusion (RRF) merges results from both searches, prioritizing high-ranking documents.


Response Generation:

Retrieved documents are synthesized into a context, and the LLM generates a cohesive answer.
Results include citations with metadata (e.g., file name, URL, source).



Why Agentic Retrieval?
This system outperforms traditional search methods by:

Breaking Down Queries: Handles multi-part questions effectively.
Parallel Execution: Runs subqueries simultaneously for efficiency.
Semantic Reranking: Ensures the most relevant results are prioritized.
Cost Efficiency: Built without Azure Foundry, using standard Azure services to reduce costs.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Report issues or suggest improvements at https://github.com/Prashant-Mouritech/Agentic-Retriever/issues.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

Azure AI Search for scalable search capabilities.
Azure OpenAI for powerful LLM and embedding models.
FastAPI for a robust API framework.

For more details, visit the repository: https://github.com/Prashant-Mouritech/Agentic-Retriever.
