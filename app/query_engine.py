import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from processor import DocumentProcessor

# Configure the Gemini API
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

class QueryEngine:
    def __init__(self):
        try:
            # Using the correct model name for gemini-2.0-flash
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",  # Updated model name
                temperature=0.3,
                convert_system_message_to_human=True
            )
            self.processor = DocumentProcessor()
            self.vector_store = None
            self.collection_name = "med_documents"
            self.persist_directory = "db"
            print("Gemini model initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            raise
        
    def load_documents(self, file_paths):
        """Process and load documents into the vector store."""
        all_chunks = []
        all_embeddings = []
        
        # Clear existing collection if it exists
        if self.vector_store is not None:
            collection = self.vector_store._collection
            if collection is not None:
                collection.delete(where={"source": {"$ne": ""}})
        
        # Process new documents
        for file_path in file_paths:
            chunks_with_embeddings = self.processor.process_document(file_path)
            chunks, embeddings = zip(*chunks_with_embeddings)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
        
        if all_chunks:  # Only create new store if we have documents
            self.vector_store = Chroma.from_texts(
                texts=all_chunks,
                embedding=self.processor.embeddings,
                metadatas=[{"source": f"doc_{i}"} for i in range(len(all_chunks))],
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        
        return len(all_chunks)
    
    def _format_docs(self, docs):
        """Format the retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question, k=3):
        """Query the document store with a question using LCEL."""
        if not self.vector_store:
            return {
                "answer": "No documents loaded. Please load documents first.",
                "sources": []
            }
        
        # Create a retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # Create the prompt template
        template = """You are a helpful medical assistant. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain using LCEL
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Get the response with error handling
        try:
            # Get the answer
            answer = chain.invoke(question)
            
            # Get source documents for reference
            docs = retriever.get_relevant_documents(question)
            
            # Format the response
            response = {
                "answer": answer,
                "sources": [str(doc.metadata) for doc in docs]
            }
            
        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            return {
                "answer": "I encountered an error while processing your request. Please try again.",
                "sources": []
            }
            
        return response

# Example usage
if __name__ == "__main__":
    engine = QueryEngine()
    # Load documents
    # engine.load_documents(["path/to/your/document.pdf"])
    # Query the engine
    # result = engine.query("What are the key points in this document?")
    # print(result)
