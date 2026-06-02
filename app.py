import os
# Tysta irriterande varningar från Hugging Face, Transformers och PyTorch i terminalen
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import base64
import re
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from database import init_db, get_all_sessions, save_message_to_db, create_new_session, get_messages_for_session, delete_session, update_session_title

# Skapa bildmappen för svinganalyser om den inte redan finns
os.makedirs("uploaded_images", exist_ok=True)

# Sätter menyn till stängd som standard
st.set_page_config(page_title="AMG Coach Pro", page_icon="🏌️‍♂️", layout="centered", initial_sidebar_state="collapsed")

# Lyxig Golf Coach Pro styling (Premium CSS-injektion)
st.markdown("""
<style>
    /* Google Font Outfit */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global App Font and Accent Headers */
    .stApp {
        font-family: 'Outfit', 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e5c483 !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar Styling (Elegant Emerald Glassmorphism) */
    section[data-testid="stSidebar"] {
        background-color: #0c1811 !important;
        border-right: 1px solid #1c3b26 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #c5a059 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Target ONLY Sidebar Buttons - No general button contamination! */
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 8px !important;
        background-color: #122b1c !important;
        color: #e5c483 !important;
        border: 1px solid #c5a059 !important;
        font-weight: 600 !important;
        transition: all 0.25s ease-in-out !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #c5a059 !important;
        color: #0c1811 !important;
        box-shadow: 0 4px 12px rgba(197, 160, 89, 0.3) !important;
        transform: translateY(-1px) !important;
        border-color: #e5c483 !important;
    }
    
    /* Target Active Chatt Button in Sidebar (primary button class) */
    section[data-testid="stSidebar"] div[data-testid="stColumn"] button[kind="primary"] {
        background-color: #c5a059 !important;
        color: #0c1811 !important;
        border: 1px solid #e5c483 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 10px rgba(197, 160, 89, 0.4) !important;
    }
    
    /* Premium Styling for source Expanders */
    .stExpander {
        background-color: #0c1811 !important;
        border: 1px solid #1c3b26 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stExpander summary {
        color: #e5c483 !important;
        font-weight: 600 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #1c3b26;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #c5a059;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏌️‍♂️ AMG-Assistenten Pro")
st.markdown("Ställ dina frågor om Athletic Motion Golf. Assistenten svarar enbart med fakta från databasen.")

init_db()

def format_docs(docs):
    formatted_texts = []
    for doc in docs:
        source = doc.metadata.get('source', 'Okänd källa')
        filename = os.path.basename(source)
        video_id = filename.replace(".txt", "")
        if video_id.startswith("transcript_"):
            video_id = video_id.replace("transcript_", "")
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
        with st.spinner("Hittade ingen databas. Bygger upp en ny med all-mpnet-base-v2..."):
            loader = DirectoryLoader("./transcripts", glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
            retriever = db.as_retriever(search_kwargs={"k": 12})
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 12})
        
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2, max_retries=1)
    
    with open("condense_prompt.txt", "r", encoding="utf-8") as f:
        condense_system_prompt = f.read()
    condense_prompt = ChatPromptTemplate.from_messages([("system", condense_system_prompt), ("placeholder", "{chat_history}"), ("human", "{input}")])
    condense_chain = condense_prompt | llm | StrOutputParser()

    def fetch_context(input_dict):
        # 1. Hämta ut enbart texten (eftersom inmatningen kan innehålla en bild)
        raw_input = input_dict["input"]
        text_query = raw_input[0]["text"] if isinstance(raw_input, list) else raw_input
        
        # 2. Skriv ALLTID om frågan för att översätta till engelska och lägga till AMG-biomekaniska sökord
        standalone_query = condense_chain.invoke({"chat_history": input_dict.get("chat_history", []), "input": text_query})
            
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

    # Pre-ladda hela den aktiva chattens historik vid uppstart (så att skärmen inte startar tomt)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.langchain_history = []
        
        db_messages = get_messages_for_session(st.session_state.current_session_id)
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

# Chatt-historiken är nu förladdad och klar vid uppstart ovan!

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            if isinstance(message["content"], list):
                # Hitta och visa bild om den finns
                img_path = None
                text_content = ""
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        img_path = item.get("image_url")
                    elif item.get("type") == "text":
                        text_content = item.get("text")
                
                if img_path and img_path != "dummy" and os.path.exists(img_path):
                    st.image(img_path, width=300)
                elif img_path == "dummy":
                    st.caption("[Bild saknas i historiken]")
                st.markdown(text_content)
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
            
            # Rendera videorna under texten i en snygg st.expander för att undvika stök
            if video_ids:
                # Filtrera bort eventuella ogiltiga IDn (YouTube IDn är alltid exakt 11 tecken långa)
                valid_vids = [v.replace("transcript_", "") for v in list(set(video_ids)) if len(v.replace("transcript_", "")) == 11]
                if valid_vids:
                    with st.expander("📺 Källvideor och instruktionsklipp (" + str(len(valid_vids)) + " st)"):
                        for clean_vid in valid_vids:
                            st.markdown(f"🔗 **[Öppna video på YouTube](https://www.youtube.com/watch?v={clean_vid})**")
                            st.video(f"https://www.youtube.com/watch?v={clean_vid}")

# --- Inmatningsfältet ---
if user_input := st.chat_input("Ställ en fråga eller ladda upp en bild...", accept_file=True, file_type=["png", "jpg", "jpeg"]):
    prompt = user_input.text
    has_image = hasattr(user_input, "files") and user_input.files
    ai_input_content = prompt
    
    if has_image:
        uploaded_image = user_input.files[0]
        
        # Generera ett unikt filnamn baserat på session_id och tidstämpel
        import time
        timestamp = int(time.time())
        img_filename = f"{st.session_state.current_session_id}_{timestamp}.png"
        img_path = os.path.join("uploaded_images", img_filename)
        
        # Spara bilden på disk
        with open(img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
            
        base64_image = get_image_base64(uploaded_image)
        if not prompt: prompt = "Analysera denna svingbild enligt AMG:s principer."
            
        with st.chat_message("user"):
            st.image(uploaded_image, width=300)
            st.markdown(prompt)
            
        content_to_save = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": img_path}]
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
                
                # Visa källvideorna i en snygg expander i realtidssvaret också
                if video_ids:
                    valid_vids = [v.replace("transcript_", "") for v in list(set(video_ids)) if len(v.replace("transcript_", "")) == 11]
                    if valid_vids:
                        with st.expander("📺 Källvideor och instruktionsklipp (" + str(len(valid_vids)) + " st)"):
                            for clean_vid in valid_vids:
                                st.markdown(f"🔗 **[Öppna video på YouTube](https://www.youtube.com/watch?v={clean_vid})**")
                                st.video(f"https://www.youtube.com/watch?v={clean_vid}")
                
                # Spara ENDAST full_response i session_state och databas (tar bort dubbelsparande)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message_to_db(st.session_state.current_session_id, "assistant", full_response)
                
                # --- BLIXTSNABB LOKAL NAMNGIVNING EFTER SVARET ---
                # Om det var första konversationen i denna chatt, döp om och ladda om
                # (Längden på st.session_state.messages är nu exakt 2 eftersom dubbelsparandet är borta!)
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