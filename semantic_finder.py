from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

miniLM_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents(folder_path="documents"):
    """Load all text files from the specified folder."""
    documents = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created {folder_path} directory. Please add text files to this folder.")
        return documents
        
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                documents[filename] = content
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return documents

def find_most_relevant_document(query, documents):
    """Find the most relevant document based on cosine similarity."""
    if not documents:
        return None, None, None
    
    # Generate embeddings for the query
    query_embedding = miniLM_embedding_model.embed_documents([query])[0]
    
    # Generate embeddings for all documents
    doc_embeddings = {}
    doc_contents = {}
    
    for filename, content in documents.items():
        doc_contents[filename] = content
        doc_embeddings[filename] = miniLM_embedding_model.embed_documents([content])[0]
    
    # Calculate cosine similarity between query and each document
    similarities = {}
    for filename, embedding in doc_embeddings.items():
        query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
        doc_embedding_reshaped = np.array(embedding).reshape(1, -1)
        similarity = cosine_similarity(query_embedding_reshaped, doc_embedding_reshaped)[0][0]
        similarities[filename] = similarity
    
    # Find the most relevant document
    if not similarities:
        return None, None, None
    
    most_relevant_file = max(similarities, key=similarities.get)
    highest_similarity = similarities[most_relevant_file]
    most_relevant_content = doc_contents[most_relevant_file]
    
    return most_relevant_file, highest_similarity, most_relevant_content

def main():
    print("Semantic Document Finder")
    print("="*26)
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        print("No documents found. Please add .txt files to the documents folder.")
        return
    
    print(f"Loaded {len(documents)} documents from the documents folder.")
    
    # Get user input
    query = input("\nEnter your query: ")
    
    # Find the most relevant document
    most_relevant_file, similarity_score, document_content = find_most_relevant_document(query, documents)
    
    # Display results
    if most_relevant_file:
        print("\nResults:")
        print(f"Most relevant document: {most_relevant_file}")
        print(f"Similarity score: {similarity_score:.4f}")
        print("\nDocument content:")
        print("-"*30)
        print(document_content)
    else:
        print("No relevant documents found.")

if __name__ == "__main__":
    main()