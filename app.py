import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# 1. Konfigurera hur webbsidan ska se ut
st.set_page_config(page_title="AMG Coach Pro", page_icon="🏌️‍♂️", layout="centered")
st.title("🏌️‍♂️ AMG-Assistenten Pro")
st.markdown("Ställ dina frågor om Athletic Motion Golf. Assistenten svarar enbart med fakta från databasen.")


# Hjälpfunktion för att formatera texten och bygga YouTube-länkar
def format_docs(docs):
    formatted_texts = []
    for doc in docs:
        # Hämta sökvägen (t.ex. transcripts\3Y5MiDlme3s.txt)
        source = doc.metadata.get('source', 'Okänd källa')
        
        # Plocka ut filnamnet och ta bort .txt för att få video-ID
        filename = os.path.basename(source)
        video_id = filename.replace(".txt", "")
        
        # Bygg länken
        youtube_link = f"https://www.youtube.com/watch?v={video_id}"
        
        # Lägg till länken precis ovanför texten från videon
        formatted_texts.append(f"KÄLLA: {youtube_link}\n{doc.page_content}")
        
    return "\n\n".join(formatted_texts)

# 2. Cacha hjärnan och databasen så de inte laddas om vid varje knapptryck
@st.cache_resource
def setup_rag_chain():
    load_dotenv()
    
    # ---------------------------------------------------------
    # DÖRRVAKTEN: Bygg databasen om den saknas i molnet
    # ---------------------------------------------------------
    if not os.path.exists("./chroma_db"):
        # Importerar bara verktygen om vi faktiskt behöver bygga databasen
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        with st.spinner("Hittade ingen databas. Bygger upp en ny från dina transkriberingar..."):
            # 1. Läs in alla .txt-filer
            loader = DirectoryLoader("./transcripts", glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            docs = loader.load()
            
            # 2. Dela upp texten
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            # 3. Skapa vektordatabasen och spara den lokalt
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
            retriever = db.as_retriever(search_kwargs={"k": 10})
    else:
        # Om mappen redan finns (som på din lokala dator), ladda den som vanligt
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # Hjärnan
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    
    # --- Mellan-hjärnan (Condense) ---
    # Öppna och läs filen för condense-prompten
    with open("condense_prompt.txt", "r", encoding="utf-8") as f:
        condense_system_prompt = f.read()
    
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    def fetch_context(input_dict):
        # Tvinga ALLA frågor genom Mellan-hjärnan så de alltid översätts till engelska
        standalone_query = condense_chain.invoke(input_dict)
        docs = retriever.invoke(standalone_query)
        return format_docs(docs)


    # Öppna och läs filen för system-prompten
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_instruction = f.read()
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    rag_chain = (
        RunnablePassthrough.assign(context=fetch_context)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Ladda kedjan i bakgrunden
rag_chain = setup_rag_chain()

# 3. Streamlits Minne (Session State)
# Om det är första gången vi besöker sidan, skapa en tom historik
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Skriv ut alla tidigare meddelanden på skärmen (för användaren)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Inmatningsfältet
if prompt := st.chat_input("Ställ en fråga om svingen..."):
    # Visa användarens meddelande direkt på skärmen
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Spara vad användaren skrev i minnet
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Skapa rutan där AI:n tänker och svarar
    with st.chat_message("assistant"):
        with st.spinner("Letar i AMG-arkivet..."):
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.langchain_history
            })
            st.markdown(response)
            
            # Kolla om AI:n har skrivit ut något ID som vi kan visa
            import re
            video_ids = re.findall(r"\[ID: ([a-zA-Z0-9_-]+)\]", response)
            
            # Visa en unik video för varje hittat ID
            for vid in list(set(video_ids)): # set() för att undvika dubbletter
                st.video(f"https://www.youtube.com/watch?v={vid}")
    
    # Spara AI:ns svar i minnet för skärmen
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Spara i LangChains specifika format för att Mellan-hjärnan ska förstå kontexten i framtiden
    st.session_state.langchain_history.append(HumanMessage(content=prompt))
    st.session_state.langchain_history.append(AIMessage(content=response))