import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser

class RAGChain:
    def __init__(self, k: int = 5, model_name: str = "gemini-2.5-pro-exp-03-25", embedding_model_name: str = "text-embedding-005"):
        # Load environment variables from .env file
        load_dotenv()

        # Initialize pinecone
        self.pc_data = {
            "pc_api_key": os.getenv("PINECONE_API_KEY"),
            "pc_env": os.getenv("PINECONE_ENVIRONMENT"),
            "pc_index_name": os.getenv("PINECONE_INDEX_NAME")
        }
        self.pinecone = Pinecone(api_key=self.pc_data["pc_api_key"], environment=self.pc_data["pc_env"])
        self.index = self.pinecone.Index(self.pc_data["pc_index_name"])

        # Initialize prompt
        self.prompt_template_string = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer as concise as possible.
        Question: {question} 
        Context: {context} 
        Answer:
        """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template_string)

        # Initilaize similarity hit count, parser
        self.k = k
        self.parser = StrOutputParser()

        # Initialize vector store, llm and embeddings
        self.embedding = VertexAIEmbeddings(model_name=embedding_model_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding)
        self.llm = ChatVertexAI(model=model_name, temperature=0.2)
        
        self.chain = self.prompt | self.llm | self.parser

    def run(self, question: str):

        # Retrieve relevant docs
        docs = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.k}).invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Get answer from LLM and fetch sources
        answer = self.chain.invoke({"question": question, "context": context})
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})
        
        return answer, sources

