import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from operator import add as add_messages


class Config:
    """Configuration management for the application."""

    def __init__(self):
        load_dotenv()
        self.pdf_path = os.getenv(
            "PDF_PATH",
            "/home/aavash/Projects/pipecat-api/graph/Stock_Market_Performance_2024.pdf",
        )
        self.persist_dir = os.getenv(
            "PERSIST_DIR", "/home/aavash/Projects/pipecat-api/persist_dir"
        )
        self.collection_name = "stock_market"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.top_k_results = 5
        self.system_prompt = """
            You are an intelligent AI assistant who answers questions about Stock Market 
            Performance in 2024 based on the PDF document loaded into your knowledge base.
            Use the retriever tool available to answer questions about the stock market 
            performance data. You can make multiple calls if needed.
            If you need to look up some information before asking a follow-up question, 
            you are allowed to do that!
            Please always cite the specific parts of the documents you use in your answers.
        """


class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(self, config: Config):
        self.config = config

    def load_and_split_documents(self) -> List:
        """Load and split PDF documents into chunks."""
        if not os.path.exists(self.config.pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {self.config.pdf_path}")

        try:
            pdf_loader = PyPDFLoader(self.config.pdf_path)
            pages = pdf_loader.load()
            print(f"✅ PDF loaded successfully. Pages: {len(pages)}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            return text_splitter.split_documents(pages)
        except Exception as e:
            raise RuntimeError(f"Failed to process documents: {e}")


class VectorStoreManager:
    """Manages the vector store creation and operations."""

    def __init__(self, config: Config, embeddings: GoogleGenerativeAIEmbeddings):
        self.config = config
        self.embeddings = embeddings

    def create_vector_store(self, documents: List) -> Chroma:
        """Create and persist a Chroma vector store."""
        if not os.path.exists(self.config.persist_dir):
            os.makedirs(self.config.persist_dir)

        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.config.persist_dir,
                collection_name=self.config.collection_name,
            )
            print("✅ Chroma vector store created.")
            return vectorstore
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")


class AgentState(TypedDict):
    """Typed dictionary representing the agent's state."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class RAGAgent:
    """Main RAG agent class orchestrating the workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.4)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        # Initialize components
        self.document_processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config, self.embeddings)

        # Process documents and create vector store
        documents = self.document_processor.load_and_split_documents()
        self.vectorstore = self.vector_store_manager.create_vector_store(documents)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.config.top_k_results}
        )

        # Setup tools and agent
        self.tools = [self._create_retriever_tool()]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.agent = self._setup_agent()

    def _create_retriever_tool(self):
        """Create the retriever tool."""

        @tool
        def retriever_tool(query: str) -> str:
            """Searches and returns relevant content from the Stock Market Performance 2024 PDF."""
            docs = self.retriever.invoke(query)
            if not docs:
                return "No relevant information found in the Stock Market Performance 2024 document."

            return "\n\n".join(
                [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
            )

        return retriever_tool

    def _should_continue(self, state: AgentState) -> bool:
        """Determine if the LLM wants to call a tool."""
        last_msg = state["messages"][-1]
        print("🧠 LLM Response:", last_msg)
        return hasattr(last_msg, "tool_calls") and last_msg.tool_calls

    def _call_llm(self, state: AgentState) -> AgentState:
        """Send messages to the LLM and get response."""
        messages = [SystemMessage(content=self.config.system_prompt)] + list(
            state["messages"]
        )
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _take_action(self, state: AgentState) -> AgentState:
        """Execute any tool calls suggested by the LLM."""
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for t in tool_calls:
            tool_name = t["name"]
            query = t["args"].get("query", "")
            print(f"🔧 Calling tool '{tool_name}' with query: {query}")

            if tool_name not in self.tools_dict:
                result = "Tool not found."
            else:
                result = self.tools_dict[tool_name].invoke(query)

            results.append(
                ToolMessage(tool_call_id=t["id"], name=tool_name, content=str(result))
            )

        return {"messages": results}

    def _setup_agent(self):
        """Configure and compile the LangGraph agent."""
        graph = StateGraph(AgentState)
        graph.add_node("llm", self._call_llm)
        graph.add_node("retriever_agent", self._take_action)
        graph.add_conditional_edges(
            "llm", self._should_continue, {True: "retriever_agent", False: END}
        )
        graph.add_edge("retriever_agent", "llm")
        graph.set_entry_point("llm")
        return graph.compile()

    def run(self):
        """Run the agent in interactive mode."""
        print("\n🤖 === RAG Agent Ready ===")
        while True:
            user_input = input("\n🧑 Your question (type 'exit' to quit): ")
            if user_input.lower() in ["exit", "quit"]:
                break

            messages = [HumanMessage(content=user_input)]

            try:
                result = self.agent.invoke({"messages": messages})
                print("\n✅ Answer:\n")
                print(result["messages"][-1].content)
            except Exception as e:
                print(f"❌ Error during response generation: {e}")


def main():
    """Entry point for the application."""
    try:
        config = Config()
        agent = RAGAgent(config)
        agent.run()
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
