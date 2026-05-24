import os
import streamlit as st
import base64
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

def get_image_base64(uploaded_file):
    image_bytes = uploaded_file.read()
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = uploaded_file.type
    return f"data:{mime_type};base64,{encoded}"

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
    if message["role"] == "user" and isinstance(message["content"], list):
        # Om det är ett multimodalt meddelande från historiken, skriv ut texten
        with st.chat_message("user"):
            st.markdown(message["content"][0]["text"])
            st.caption("[Innehåller en uppladdad bild]")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 4. Inmatningsfältet
# 4. Inmatningsfältet (nu med inbyggd bilduppladdning!)
# Vi lägger till accept_file=True och file_type i funktionen
if user_input := st.chat_input("Ställ en fråga eller ladda upp en bild...", accept_file=True, file_type=["png", "jpg", "jpeg"]):
    
    prompt = user_input.text
    has_image = hasattr(user_input, "files") and user_input.files
    
    # Förbered vad vi ska skicka till AI:n
    ai_input_content = prompt
    
    # 1. Om användaren laddade upp en bild
    if has_image:
        uploaded_image = user_input.files[0]
        base64_image = get_image_base64(uploaded_image)
        
        # Om användaren laddade upp en bild men inte skrev någon text, skapa en default-text
        if not prompt:
            prompt = "Analysera denna svingbild enligt AMG:s principer."
            
        with st.chat_message("user"):
            st.image(uploaded_image, width=300)
            st.markdown(prompt)
            
        # Spara i Session State (så det visas nästa gång sidan laddar)
        # Vi sparar inte base64-bilden i 'messages' för UI-historiken för att spara prestanda, bara en indikation
        st.session_state.messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": "dummy"}] 
        })
        
        # Bygg LangChain-formatet för multimodal data
        ai_input_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": base64_image}}
        ]
        
    # 2. Om det bara var text
    elif prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Kör igång AI-hjärnan om det fanns input (text eller bild)
    if prompt or has_image:
        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            
            tänker_html = """
            <style>
                .pulsing-text {
                    animation: pulse 1.5s infinite;
                    color: #888;
                    font-style: italic;
                }
                @keyframes pulse {
                    0% { opacity: 0.4; }
                    50% { opacity: 1; }
                    100% { opacity: 0.4; }
                }
            </style>
            <div class="pulsing-text">Tänker...</div>
            """
            text_placeholder.markdown(tänker_html, unsafe_allow_html=True)
            
            full_response = ""
            
            # Skicka in ai_input_content istället för bara 'prompt'
            stream = rag_chain.stream({
                "input": ai_input_content,
                "chat_history": st.session_state.langchain_history
            })
            
            for chunk in stream:
                full_response += chunk
                
                if "[ID:" in full_response:
                    display_text = full_response.split("[ID:")[0]
                else:
                    display_text = full_response
                    
                text_placeholder.markdown(display_text + "▌")
                
            text_placeholder.markdown(display_text)
            
            import re
            video_ids = re.findall(r"\[ID: ([a-zA-Z0-9_-]+)\]", full_response)
            clean_response = re.sub(r"\[ID: [a-zA-Z0-9_-]+\]", "", full_response)
            
            for vid in list(set(video_ids)):
                st.video(f"https://www.youtube.com/watch?v={vid}")
                
        st.session_state.messages.append({"role": "assistant", "content": clean_response})
        
        # Spara till LangChain History (vi sparar textversionen av prompten i historiken, att spara hela Base64 strängen sabbar ofta minnet för framtida frågor)
        st.session_state.langchain_history.append(HumanMessage(content=prompt))
        st.session_state.langchain_history.append(AIMessage(content=full_response))