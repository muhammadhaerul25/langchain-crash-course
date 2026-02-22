# Step 1: Load the knowledge base
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Step 2: Load JSON file
print("\n--- Step 2: Loading JSON File ---")
loader = JSONLoader(
    file_path=str(BASE_DIR / "documents" / "transaksi.json"),
    jq_schema=".[]",
    text_content=False,
)
documents = loader.load()
# print(documents)


# Step 3: Chunk the documents 
print("\n--- Step 3: Chunking Docs ---")
from langchain_text_splitters import RecursiveJsonSplitter
splitter = RecursiveJsonSplitter(max_chunk_size=300)

with open(BASE_DIR / "documents" / "transaksi.json", "r", encoding="utf-8") as file:
    transaksi_data = json.load(file)

json_chunks = splitter.split_json(json_data={"transaksi": transaksi_data})
chunk_documents = [
    Document(page_content=json.dumps(chunk, ensure_ascii=False)) for chunk in json_chunks
]
# for chunk in json_chunks[:3]:
#     print(chunk)

# Step 4: Load model embeddings
print("\n--- Step 4: Loading Model Embeddings ---")
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

model_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
db_dir = "db"
persistent_directory = "db/db_json_data_2"

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

create_vector_store(chunk_documents, model_embeddings, "db_json_data_2")


# Step 5: Load vector store and test similarity search
# Function to query a vector store
print("\n--- Step 5: Querying Vector Store ---")
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
            search_kwargs={"k": 3},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        # print(f"\n--- Relevant Documents for {store_name} ---")
        # for i, doc in enumerate(relevant_docs, 1):
        #     print(f"Document {i}:\n{doc.page_content}\n")
        #     if doc.metadata:
        #         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")



# Step 6: Test the RAG pipeline with RetrievalQA
# Define the user's question
print("\n--- Step 6: Testing RAG Pipeline ---")
query = "Berikan harga termahal dan kuantity dari spesifikasi semen pcc indosemen, pada tahun berapa, proyeknya apa dan di provinsi mana?"
# query = "Berapa jenis tipe bangunan?"

# Retrieve relevant documents based on the query
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=model_embeddings,
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)

# 3 type of retriever
# 1. similairty -> mengembalikan dokumen yang paling mirip dengan query berdasarkan embedding (the most relevant)
# 2. mmr -> mengembalikan dokumen yang paling relevan dengan query berdasarkan embedding, tetapi juga mempertimbangkan keragaman dokumen yang dikembalikan (diverser)
# 3. thersholde -> ada batas kerelevanan dokument

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








