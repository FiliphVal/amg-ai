import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def start_chat():
    load_dotenv()
    print("Väcker databasen och AI-modellen...")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # Vi använder den modernaste modellen
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    
    # ---------------------------------------------------------
    # STEG 1: MELLAN-HJÄRNAN (Skriver om otydliga frågor)
    # ---------------------------------------------------------
    with open("condense_prompt.txt", "r", encoding="utf-8") as f:
        condense_system_prompt = f.read()
    
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    # Kedjan för att skriva om frågan
    condense_chain = condense_prompt | llm | StrOutputParser()

    # Denna funktion styr logiken innan vi slår mot databasen
    def fetch_context(input_dict):
        chat_history = input_dict.get("chat_history", [])
        original_query = input_dict["input"]
        
        # Om vi har pratat tidigare, be AI:n städa upp frågan först
        if len(chat_history) > 0:
            standalone_query = condense_chain.invoke(input_dict)
            print(f"   [Systemet optimerade sökningen till: '{standalone_query}']")
            docs = retriever.invoke(standalone_query)
        else:
            # Första frågan i chatten behöver inte skrivas om
            docs = retriever.invoke(original_query)
            
        return format_docs(docs)

    # ---------------------------------------------------------
    # STEG 2: HUVUD-HJÄRNAN (Svarar på frågan)
    # ---------------------------------------------------------
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_instruction = f.read()
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    # Den slutgiltiga RAG-kedjan
    rag_chain = (
        RunnablePassthrough.assign(context=fetch_context)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n" + "="*50)
    print("AMG-ASSISTENTEN PRO ÄR REDO! (Skriv 'avsluta' för att stänga)")
    print("="*50)
    
    chat_history = []
    
    while True:
        user_input = input("\nDin fråga: ")
        
        if user_input.lower() in ['avsluta', 'quit', 'exit']:
            print("Stänger ner assistenten. Bra jobbat idag!")
            break
            
        print("Funderar och letar i arkivet...\n")
        
        # Kör hela kedjan: Skriv om -> Sök -> Svara
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        print("\n" + response)
        
        # Spara i minnet för framtida följdfrågor
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    start_chat()