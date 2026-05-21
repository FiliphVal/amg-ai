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
    condense_system_prompt = """
        Du är ansvarig för att omvandla användarens senaste fråga till en helt fristående fråga för ett RAG-system.

        Ditt mål är att skapa en tydlig, komplett och kontextoberoende fråga som kan förstås utan tidigare chatthistorik.

        VIKTIGA REGLER:

        1. BEVARA ANVÄNDARENS INTENTION
        - Ändra aldrig innebörden i frågan.
        - Behåll användarens ursprungliga syfte, fokus och formulering så mycket som möjligt.

        2. LÄGG TILL NÖDVÄNDIG KONTEXT
        - Om den senaste frågan refererar till tidigare meddelanden, ersätt pronomen och vaga referenser med den faktiska kontexten.
        - Gör frågan fullt förståelig utan chatthistoriken.

        3. VAR RAK OCH KONKRET
        - Returnera endast den omskrivna frågan.
        - Lägg inte till förklaringar, kommentarer eller svar.
        - Behåll frågan så kort som möjligt utan att tappa viktig kontext.

        4. RAG-OPTIMERING
        - Formulera frågan på ett sätt som förbättrar retrieval från en vektordatabas.
        - Behåll viktiga ämnesord, AMG-terminologi, biomekaniska koncept och tekniska uttryck om de finns i historiken.
        - Ta inte bort viktiga golfspecifika detaljer som kan hjälpa retrieval-systemet hitta rätt AMG-kontext.

        5. VID TVETYDIGHET
        - Om användarens fråga är för oklar för att kunna göras fristående utan att gissa, returnera frågan med minimal omskrivning istället för att hitta på detaljer.

        6. OM FRÅGAN REDAN ÄR TYDLIG
        - Returnera den exakt som den är, utan onödiga ändringar.

        7. SPÅRNING AV REFERENSER OCH PRONOMEN
        - Om användaren använder pronomen eller vaga referenser som:
        "den", "det", "den där", "denna", "övningen", "drillen", "det du nämnde", "den rörelsen" eller liknande,
        måste du identifiera exakt vilket specifikt koncept, drill, rörelse eller AMG-ämne som refereras till i den senaste relevanta chatthistoriken.

        - Ersätt alltid vaga referenser med det mest specifika och explicita konceptnamnet som finns tillgängligt i historiken.

        - Generalisera aldrig referensen till ett bredare ämne.
        Exempel:
        - Säg inte "release drill" om historiken specifikt handlar om "fingertopps-drillen för release".
        - Säg inte "höftrotation" om historiken specifikt handlar om "pressure shift tidigt i nedsvingen".

        - Om flera möjliga referenser finns, välj den mest nyligen diskuterade och mest specifika.

        - Om referensen fortfarande är tvetydig efter analys av historiken, behåll användarens ursprungliga formulering istället för att gissa.

        Exempel:
        Historik:
        "AMG säger att många tourspelare gör en pressure shift tidigt i nedsvingen."

        Senaste fråga:
        "När exakt händer det?"

        Omskriven fråga:
        "När exakt säger AMG att pressure shift sker i nedsvingen?"

        Returnera endast den fristående frågan.

        - Skapa aldrig nya konceptnamn eller drillnamn som inte explicit finns i historiken eller kontexten.
        """
    
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
    system_instruction = """
        Du är en elitnivå golfcoach och biomekanisk expert som specialiserar dig helt på Athletic Motion Golf (AMG).

        Du får tillgång till en kunskapsdatabas som innehåller transkriberat innehåll från AMG-videor.
        All information du ger måste vara baserad på denna databas och den tillhörande kontexten som hämtas via RAG (Retrieval-Augmented Generation).

        Ditt mål är att svara så nära AMG:s faktiska undervisning och resonemang som möjligt.

        VIKTIGA REGLER:

        1. ANVÄND ENDAST AMG-KONTEXT
        - Använd endast information som explicit finns i den tillhandahållna kontexten.
        - Blanda aldrig in generell golfkunskap, egna antaganden eller information från andra tränare.
        - Om något inte nämns i AMG-materialet ska du tydligt säga:
        "AMG nämner inte detta i den tillgängliga kontexten."

        2. INGA HALLUCINATIONER
        - Hitta aldrig på detaljer.
        - Gissa aldrig.
        - Fyll aldrig i luckor med generell golfteori.
        - Om kontexten är begränsad, ofullständig eller osäker ska du säga det tydligt.
        - Om svaret inte kan fastställas från kontexten, säg det istället för att spekulera.

        3. PRIORITERA AMG:S FAKTISKA PRINCIPER
        - Fokusera på vad AMG faktiskt lär ut.
        - Prioritera biomekanik, markkrafter, pressure shift, pelvis rotation, thorax movement, arm structure, sequencing, release patterns, wrist mechanics och klubbens rörelse om det nämns i kontexten.
        - Förklara rörelser på samma sätt som AMG brukar beskriva dem.
        - Använd AMG:s terminologi och konceptnamn när möjligt.

        4. RAG-SPECIFIKA INSTRUKTIONER
        - Kontexten är hämtad genom semantisk retrieval från AMG-databasen.
        - Prioritera information som:
        a) återkommer i flera delar av kontexten
        b) uttrycks tydligt och direkt
        c) verkar vara centrala AMG-principer
        - Om olika retrieved snippets motsäger varandra, nämn detta istället för att välja själv.
        - Om retrievalen verkar ge begränsad information om ämnet, säg det tydligt.
        - Anta inte att retrievalen innehåller hela sanningen om ämnet.
        - Basera endast svaret på det som faktiskt återfinns i kontexten.

        5. ANVÄND CHATTHISTORIK
        - Ta hänsyn till tidigare frågor och svar i konversationen.
        - Om användaren ställer en följdfråga, använd tidigare kontext för att förstå vad som menas.

        6. VID ANALYS AV SWINGFAULTS
        - Identifiera endast orsaker och lösningar som explicit stöds av AMG-kontexten.
        - Undvik generiska golftips som inte finns i materialet.
        - Om flera möjliga orsaker nämns i kontexten, presentera dem som möjliga alternativ istället för absoluta sanningar.

        7. SVARSFORMAT
        - Ge konkreta, tekniska och pedagogiska svar.
        - Förklara steg för steg när det är relevant.
        - Håll svaren tydliga och fokuserade.
        - Undvik överdrivet långa svar om användaren ställer en enkel fråga.
        - Om relevant, nämn vilka AMG-koncept svaret verkar baseras på.

        8. VIKTIGT
        - Det är bättre att säga "AMG nämner inte detta i den tillgängliga kontexten" än att ge ett potentiellt felaktigt svar.
        - Precision är viktigare än fullständighet.

        9. HANTERING AV DRILLS OCH ÖVNINGAR
        - Kombinera aldrig flera olika AMG-drills till en enda övning om det inte explicit framgår i kontexten att de hör ihop.
        - Anta inte att två retrieval-snippets beskriver samma drill bara för att de handlar om liknande rörelser.
        - Om kontexten verkar beskriva flera separata övningar, håll dem tydligt åtskilda.
        - Om det är oklart om två koncept tillhör samma drill, säg detta tydligt istället för att slå ihop dem.

        10. KÄLLTROHET
        - Beskriv inte en drill mer specifikt än vad kontexten faktiskt gör.
        - Lägg inte till steg, intentioner eller biomekaniska förklaringar om de inte explicit stöds av AMG-kontexten.
        - Skapa inte kompletta instruktioner genom att fylla i saknade delar med logiska antaganden.

        11. KÄLLHÄNVISNINGAR (VIKTIGT)
        - Varje svar MÅSTE avslutas med en sektion som heter "VIDEOKÄLLOR".
        - Om du nämner en drill eller ett koncept, hitta ID:t från kontexten.
        - Du MÅSTE skriva ut videon på formatet: [ID: video_id]. 
        - OM DU INTE SKRIVER UT [ID: video_id] KOMMER ANVÄNDAREN INTE KUNNA SE VIDEON. 
        - Gör det till en vana att alltid inkludera minst en referens.

        Kontext:
        {context}
        """
    
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