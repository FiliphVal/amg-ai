import os
import streamlit as st
import base64
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from database import init_db, get_all_sessions, save_message_to_db, create_new_session, get_messages_for_session, delete_session, update_session_title

# Sätter menyn till stängd som standard
st.set_page_config(page_title="AMG Coach Pro", page_icon="🏌️‍♂️", layout="centered", initial_sidebar_state="collapsed")
st.title("🏌️‍♂️ AMG-Assistenten Pro")
st.markdown("Ställ dina frågor om Athletic Motion Golf. Assistenten svarar enbart med fakta från databasen.")

init_db()

def format_docs(docs):
    formatted_texts = []
    for doc in docs:
        source = doc.metadata.get('source', 'Okänd källa')
        filename = os.path.basename(source)
        video_id = filename.replace(".txt", "")
        youtube_link = f"https://www.youtube.com/watch?v={video_id}"
        formatted_texts.append(f"KÄLLA: {youtube_link}\n{doc.page_content}")
    return "\n\n".join(formatted_texts)

def get_image_base64(uploaded_file):
    image_bytes = uploaded_file.read()
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = uploaded_file.type
    return f"data:{mime_type};base64,{encoded}"

@st.cache_resource
def setup_rag_chain():
    load_dotenv()
    if not os.path.exists("./chroma_db"):
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        with st.spinner("Hittade ingen databas. Bygger upp en ny..."):
            loader = DirectoryLoader("./transcripts", glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
            retriever = db.as_retriever(search_kwargs={"k": 10})
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 10})
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2, max_retries=1)
    
    with open("condense_prompt.txt", "r", encoding="utf-8") as f:
        condense_system_prompt = f.read()
    condense_prompt = ChatPromptTemplate.from_messages([("system", condense_system_prompt), ("placeholder", "{chat_history}"), ("human", "{input}")])
    condense_chain = condense_prompt | llm | StrOutputParser()

    def fetch_context(input_dict):
        # 1. Hämta ut enbart texten (eftersom inmatningen kan innehålla en bild)
        raw_input = input_dict["input"]
        text_query = raw_input[0]["text"] if isinstance(raw_input, list) else raw_input
        
        # 2. Är chatten helt ny? Slösa INTE ett API-anrop på Mellan-hjärnan!
        if not input_dict.get("chat_history"):
            standalone_query = text_query
        else:
            # Bara om vi har tidigare historik måste AI:n skriva om frågan
            standalone_query = condense_chain.invoke({"chat_history": input_dict["chat_history"], "input": text_query})
            
        docs = retriever.invoke(standalone_query)
        return format_docs(docs)

    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_instruction = f.read()
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_instruction), ("placeholder", "{chat_history}"), ("human", "{input}")])
    
    rag_chain = (RunnablePassthrough.assign(context=fetch_context) | qa_prompt | llm | StrOutputParser())
    return rag_chain, llm

rag_chain, llm = setup_rag_chain()

# --- Popups för att Byta namn / Ta bort chatt ---
@st.dialog("Byt namn på chatt")
def rename_dialog(sess_id, current_title):
    new_title = st.text_input("Nytt namn:", value=current_title)
    if st.button("Spara", use_container_width=True):
        update_session_title(sess_id, new_title)
        st.rerun()

@st.dialog("Ta bort chatt")
def delete_dialog(sess_id):
    st.warning("Är du säker? Detta kan inte ångras.")
    if st.button("🗑️ Ja, ta bort", type="primary", use_container_width=True):
        delete_session(sess_id)
        if st.session_state.current_session_id == sess_id:
            del st.session_state.current_session_id
        st.rerun()

# --- Sidomeny & Hantering av chattrådar ---
with st.sidebar:
    st.header("🏌️‍♂️ Tidigare svingtankar")
    if st.button("➕ Ny chatt", type="primary", use_container_width=True):
        st.session_state.current_session_id = create_new_session()
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.rerun()
        
    st.divider()
    saved_sessions = get_all_sessions()
    
    if "current_session_id" not in st.session_state:
        if saved_sessions:
            st.session_state.current_session_id = saved_sessions[0][0]
        else:
            st.session_state.current_session_id = create_new_session()
            saved_sessions = get_all_sessions()

    # Kompakt loop för chattar med 3-prickars meny (Popover)
    # Kompakt loop för chattar med 3-prickars meny (Popover)
    for sess_id, sess_title in saved_sessions:
        col1, col2 = st.columns([8, 2])
        
        # Kontrollera om detta är den aktiva chatten
        is_active = (sess_id == st.session_state.current_session_id)
        button_label = f"💬 {sess_title}" if not is_active else f"👉 {sess_title}"
        
        # Om den är aktiv, ge den accentfärgen (primary), annars standard (secondary)
        btn_type = "primary" if is_active else "secondary"
        
        # 1. Huvudknappen för att välja chatten
        if col1.button(button_label, key=f"sel_{sess_id}", use_container_width=True, type=btn_type):
            st.session_state.current_session_id = sess_id
            
            db_messages = get_messages_for_session(sess_id)
            st.session_state.messages = []
            st.session_state.langchain_history = []
            
            for role, content in db_messages:
                if content.startswith("[") and "image_url" in content:
                    import ast
                    try: parsed_content = ast.literal_eval(content)
                    except: parsed_content = content
                else: parsed_content = content
                
                st.session_state.messages.append({"role": role, "content": parsed_content})
                
                if role == "user":
                    text_prompt = parsed_content[0]["text"] if isinstance(parsed_content, list) else parsed_content
                    st.session_state.langchain_history.append(HumanMessage(content=text_prompt))
                else:
                    st.session_state.langchain_history.append(AIMessage(content=parsed_content))
            st.rerun()
            
        # 2. Popover-menyn för hantering
        with col2.popover("⋮"):
            if st.button("Ändra namn", key=f"ren_{sess_id}", use_container_width=True):
                rename_dialog(sess_id, sess_title)
            if st.button("Ta bort", key=f"del_{sess_id}", use_container_width=True):
                delete_dialog(sess_id)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            if isinstance(message["content"], list):
                st.markdown(message["content"][0]["text"])
                st.caption("[Innehåller en uppladdad bild]")
            else:
                st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            content = message["content"]
            
            # Leta efter video-taggar i historiken
            video_ids = re.findall(r"\[ID:\s*([a-zA-Z0-9_-]+)\]", content)
            clean_content = re.sub(r"\[ID:\s*[a-zA-Z0-9_-]+\]", "", content)
            
            # Skriv ut texten utan taggarna
            st.markdown(clean_content)
            
            # Rendera videorna under texten
            for vid in list(set(video_ids)):
                st.video(f"https://www.youtube.com/watch?v={vid}")

# --- Inmatningsfältet ---
if user_input := st.chat_input("Ställ en fråga eller ladda upp en bild...", accept_file=True, file_type=["png", "jpg", "jpeg"]):
    prompt = user_input.text
    has_image = hasattr(user_input, "files") and user_input.files
    ai_input_content = prompt
    
    if has_image:
        uploaded_image = user_input.files[0]
        base64_image = get_image_base64(uploaded_image)
        if not prompt: prompt = "Analysera denna svingbild enligt AMG:s principer."
            
        with st.chat_message("user"):
            st.image(uploaded_image, width=300)
            st.markdown(prompt)
            
        content_to_save = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": "dummy"}]
        st.session_state.messages.append({"role": "user", "content": content_to_save})
        save_message_to_db(st.session_state.current_session_id, "user", content_to_save)
        
        ai_input_content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": base64_image}}]
        
    elif prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message_to_db(st.session_state.current_session_id, "user", prompt)

    # Kör AI om det fanns input
    if prompt or has_image:
        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            tänker_html = """
            <style>
                .pulsing-text { animation: pulse 1.5s infinite; color: #888; font-style: italic; }
                @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
            </style><div class="pulsing-text">Tänker...</div>
            """
            text_placeholder.markdown(tänker_html, unsafe_allow_html=True)
            
            try:
                full_response = ""
                stream = rag_chain.stream({"input": ai_input_content, "chat_history": st.session_state.langchain_history})
                
                for chunk in stream:
                    full_response += chunk
                    display_text = full_response.split("[ID:")[0] if "[ID:" in full_response else full_response
                    text_placeholder.markdown(display_text + "▌")
                    
                text_placeholder.markdown(display_text)
                
                import re
                video_ids = re.findall(r"\[ID: ([a-zA-Z0-9_-]+)\]", full_response)
                clean_response = re.sub(r"\[ID: [a-zA-Z0-9_-]+\]", "", full_response)
                
                for vid in list(set(video_ids)):
                    st.video(f"https://www.youtube.com/watch?v={vid}")
                    
                st.session_state.messages.append({"role": "assistant", "content": clean_response})
                save_message_to_db(st.session_state.current_session_id, "assistant", clean_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message_to_db(st.session_state.current_session_id, "assistant", full_response)
                
                # --- BLIXTSNABB LOKAL NAMNGIVNING EFTER SVARET ---
                # Om det var första konversationen i denna chatt, döp om och ladda om
                if len(st.session_state.messages) == 2:
                    words = prompt.split()
                    short_title = " ".join(words[:3])
                    if len(words) > 3:
                        short_title += "..."
                    update_session_title(st.session_state.current_session_id, short_title)
                    st.rerun() # Nu är det säkert att ladda om sidan!
                        
            except Exception as e:
                text_placeholder.empty()
                if "429" in str(e):
                    st.error("⏳ För många frågor! Googles gratistak är nått. Vänta några sekunder innan du provar igen.")
                else:
                    st.error(f"🚨 Ett fel uppstod: {e}")