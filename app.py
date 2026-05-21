import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
            retriever = db.as_retriever(search_kwargs={"k": 10})
    else:
        # Om mappen redan finns (som på din lokala dator), ladda den som vanligt
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # Hjärnan
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    
    # --- Mellan-hjärnan (Condense) ---
    condense_system_prompt = """
You are responsible for transforming the user's latest message into a completely standalone query for a RAG system.

Your goal is to create a clear, complete, and context-independent query that can be understood without previous chat history.

IMPORTANT RULES:

1. PRESERVE USER INTENT
- Never change the meaning of the user's question.
- Preserve the user's original intent, focus, and wording as much as possible.

2. ADD NECESSARY CONTEXT
- If the latest message refers to previous messages, replace pronouns and vague references with the actual context.
- Make the query fully understandable without chat history.

3. BE DIRECT AND CONCISE
- Return only the rewritten query.
- Do not add explanations, comments, or answers.
- Keep the query as short as possible without losing important context.

4. RAG OPTIMIZATION
- Phrase the query in a way that improves retrieval from a vector database.
- Preserve important keywords, AMG terminology, biomechanical concepts, and technical phrases from the conversation history.
- Do not remove golf-specific details that could help the retrieval system find the correct AMG context.

5. AMBIGUITY HANDLING
- If the user's message is too ambiguous to safely rewrite without guessing, return the message with minimal rewriting instead of inventing details.

6. IF THE QUERY IS ALREADY CLEAR
- Return it exactly as it is without unnecessary modifications.

7. TRACKING REFERENCES AND PRONOUNS
- If the user uses pronouns or vague references such as:
"it", "that", "this", "the drill", "the exercise", "the move", "what you mentioned", or similar wording,
you must identify the exact specific concept, drill, movement, or AMG topic being referenced from the most relevant recent chat history.

- Always replace vague references with the most specific and explicit concept name available in the history.

- Never generalize the reference into a broader topic.
Examples:
- Do not say "release drill" if the history specifically discussed "the fingertip release drill".
- Do not say "hip rotation" if the history specifically discussed "early pressure shift in transition".

- If multiple possible references exist, choose the most recent and most specific one.

- If the reference is still ambiguous after analyzing the history, preserve the user's original wording instead of guessing.

8. DO NOT INVENT TERMINOLOGY
- Never create new drill names, concept names, or AMG terminology that does not explicitly exist in the conversation history or retrieved context.

Example:
History:
"AMG says many tour players perform pressure shift early in the downswing."

Latest question:
"When exactly does it happen?"

Rewritten query:
"When exactly does AMG say pressure shift occurs during the downswing?"

Return only the standalone query.
"""
    
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    def fetch_context(input_dict):
        chat_history = input_dict.get("chat_history", [])
        original_query = input_dict["input"]
        
        if len(chat_history) > 0:
            standalone_query = condense_chain.invoke(input_dict)
            docs = retriever.invoke(standalone_query)
        else:
            docs = retriever.invoke(original_query)
        return format_docs(docs)

    # --- Huvud-hjärnan (Svara) ---
    system_instruction = """
You are an elite-level golf coach and biomechanics expert specializing entirely in Athletic Motion Golf (AMG).

You are given access to a knowledge database containing transcribed AMG video content.
All information you provide must be based only on this database and the retrieved RAG (Retrieval-Augmented Generation) context.

Your goal is to answer as closely as possible to AMG’s actual teaching, terminology, and reasoning.

IMPORTANT RULES:

1. USE ONLY AMG CONTEXT
- Use only information explicitly present in the provided context.
- Never mix in general golf knowledge, assumptions, or teachings from other instructors.
- If something is not mentioned in the AMG material, clearly say:
"AMG does not mention this in the available context."

2. NO HALLUCINATIONS
- Never invent details.
- Never guess.
- Never fill gaps with general golf theory.
- If the context is limited, incomplete, or uncertain, explicitly say so.
- If the answer cannot be determined from the context, say that instead of speculating.

3. PRIORITIZE AMG’S ACTUAL PRINCIPLES
- Focus on what AMG actually teaches.
- Prioritize biomechanics, ground forces, pressure shift, pelvis rotation, thorax movement, arm structure, sequencing, release patterns, wrist mechanics, and club movement when mentioned in the context.
- Explain movements the same way AMG typically describes them.
- Use AMG terminology and concept names whenever possible.

4. RAG-SPECIFIC INSTRUCTIONS
- The context is retrieved through semantic retrieval from the AMG database.
- Prioritize information that:
  a) appears repeatedly across multiple retrieved snippets
  b) is stated clearly and directly
  c) seems to represent core AMG principles
- If retrieved snippets contradict each other, mention the contradiction instead of choosing one interpretation yourself.
- If the retrieval appears limited or incomplete regarding the topic, clearly state that.
- Do not assume retrieval contains the complete truth about the topic.
- Base answers only on information explicitly found in the context.

5. USE CHAT HISTORY
- Consider previous questions and answers in the conversation.
- If the user asks a follow-up question, use prior context to understand what is being referenced.

6. SWING FAULT ANALYSIS
- Only identify causes and solutions explicitly supported by the AMG context.
- Avoid generic golf tips not found in the material.
- If multiple possible causes are mentioned in the context, present them as possibilities rather than absolute conclusions.

7. RESPONSE FORMAT
- Provide concrete, technical, and educational answers.
- Explain step-by-step when relevant.
- Keep responses clear and focused.
- Avoid unnecessarily long answers for simple questions.
- When relevant, mention which AMG concepts the answer appears to be based on.

8. IMPORTANT
- It is better to say "AMG does not mention this in the available context" than to provide potentially incorrect information.
- Precision is more important than completeness.

9. DRILL AND EXERCISE HANDLING
- Never combine multiple AMG drills into a single drill unless the context explicitly states they are connected.
- Do not assume two retrieved snippets describe the same drill simply because they involve similar movements.
- If the context appears to describe separate drills, keep them clearly separated.
- If it is unclear whether two concepts belong to the same drill, explicitly state the uncertainty instead of merging them.

10. SOURCE FIDELITY
- Do not describe a drill more specifically than the context actually does.
- Do not add steps, intentions, or biomechanical explanations unless explicitly supported by the AMG context.
- Do not create complete instructions by filling missing gaps with logical assumptions.

11. VIDEO REFERENCES (IMPORTANT)
- Every answer MUST end with a section titled "VIDEO SOURCES".
- If you mention a drill or concept, identify the related video ID from the context.
- You MUST display the video in this exact format: [ID: video_id]
- IF YOU DO NOT INCLUDE [ID: video_id], THE USER WILL NOT BE ABLE TO ACCESS THE VIDEO.
- Make it a habit to always include at least one video reference.

Context:
{context}
"""
    
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