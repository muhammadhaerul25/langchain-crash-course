# Step 1: Load the knowledge base
import os
import dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
dotenv.load_dotenv()

# baca file path
file_path = "tutorial/documents/profile_perusahaan.txt"
# buat object loader
loader = TextLoader(file_path)
documents = loader.load()
# print(documents)


# #--------------------------------------------------------------------#
# Step 2: Chunk the documents
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
docs_split = text_splitter.split_documents(documents)
# print(docs_split)


# #--------------------------------------------------------------------#
# # Step 3: Load model embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
model_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# #--------------------------------------------------------------------#
# # Step 4: Create vector store with Chroma DB
from langchain_community.vectorstores import Chroma
persistent_directory = "db/chroma_db_google_embeddings"
db_dir = "db"

def create_vector_store(docs, model_embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, model_embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

create_vector_store(docs_split, model_embeddings, "chroma_db_google_embeddings")


#--------------------------------------------------------------------#
# Step 5: Load vector store and test similarity search
# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# # Step 6: Test the vector store with a query
# user_query = "Apa saja produk yang ditawarkan oleh company ini?"
# query_vector_store("chroma_db_google_embeddings", user_query, model_embeddings)
# print("\n--- Finished querying the vector store ---")

# Step 7: Test the RAG pipeline with RetrievalQA
# Define the user's question
query = "Apakah perusahaan ini adalah BUMN?"

# Retrieve relevant documents based on the query
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=model_embeddings,
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create a ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)