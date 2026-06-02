import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vector_db():
    print("1. Letar efter textfiler i mappen 'transcripts'...")
    
    # Laddar in alla filer. Vi tvingar utf-8 kodning så att åäö och specialtecken inte kraschar programmet
    loader = DirectoryLoader(
        "transcripts", 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    print(f"-> Laddade in {len(documents)} dokument.\n")

    print("2. Hackar upp texterna i mindre bitar (Chunks)...")
    # Vi klipper texten i bitar om ca 1000 tecken, med 200 teckens överlappning
    # Överlappningen gör att vi inte råkar klippa en viktig mening mitt i.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Skapade totalt {len(chunks)} text-bitar.\n")

    print("3. Laddar in AI-modellen för att omvandla text till matematik...")
    # Laddar ner en liten, snabb och gratis modell från HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print("\n4. Bygger Vektordatabasen (Detta tar lite tid, ha tålamod!)...")
    # Skapar databasen och sparar den permanent i en mapp som heter "chroma_db"
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("\n" + "*"*40)
    print("SUCCÉ! Databasen är färdig och sparad i mappen 'chroma_db'.")
    print("*"*40)

if __name__ == "__main__":
    create_vector_db()