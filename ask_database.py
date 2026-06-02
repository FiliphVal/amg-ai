import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def search_amg_database(query):
    # 1. Ladda in samma matematiska modell som vi byggde databasen med
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # 2. Koppla upp oss mot den befintliga databasen
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 3. Gör en sökning och hämta de 3 mest relevanta textbitarna
    print(f"\nSöker i databasen efter: '{query}'...\n")
    results = db.similarity_search(query, k=3)
    
    # 4. Skriv ut resultaten snyggt
    print("="*50)
    print("TOPP 3 TRÄFFAR I AMG-DATABASEN:")
    print("="*50)
    
    for i, doc in enumerate(results, start=1):
        print(f"\nTRÄFF {i}:")
        
        # Hämtar den råa sökvägen (t.ex. transcripts\3Y5MiDlme3s.txt)
        source = doc.metadata.get('source', 'Okänd fil')
        
        # 1. Plocka ut enbart filnamnet från sökvägen
        filename = os.path.basename(source)
        
        # 2. Sudda ut filändelsen så vi bara har ID:t kvar
        video_id = filename.replace(".txt", "")
        
        # 3. Bygg ihop den slutgiltiga länken
        youtube_link = f"https://www.youtube.com/watch?v={video_id}"
        
        # Skriv ut resultatet
        print(f"Källa: {youtube_link}")
        print("-" * 20)
        print(doc.page_content)
        print("-" * 20)

        
if __name__ == "__main__":
    # Här kan du ändra frågan till vad som helst om svingen!
    test_question = "how to shallow the club in the downswing"
    search_amg_database(test_question)