import asyncio
from playwright.async_api import async_playwright

from controller import   Database2 , Gemini_ask, Navigater
import os 
import dotenv

parent_dir = os.path.dirname(__file__)

dotenv.load_dotenv(dotenv_path=os.path.join(parent_dir, ".env"))


#asyncio.run(Navigater().run())


def createRag():
    path = r"C:\Users\Hasina_IA\Documents\andy\andy\document.json"
    db = Database2(path=path)
    db.load_data()
    db.create_embedding()

    # Poser une question


def geminichat():
    chat = Gemini_ask(
        prompt_type="assistant"
        
    )
    response = chat.conversation(user_input="salut")
    print(response)
    
    
#geminichat()

def testRag():    
    path = r"C:\Users\Hasina_IA\Documents\andy\andy\document.json"
    db = Database2(path , prompt_type="generation")
    # 1️⃣ Charger et découper les documents
    #db.load_data()
    print("📄 Documents chargés et découpés.")

    # 2️⃣ Créer les embeddings et index FAISS
    #db.create_embedding()
    print("🧠 Embeddings créés et index FAISS sauvegardé.")
    db.load_index()
    # 3️⃣ Créer la chaîne RAG
    db.create_rag()
    print("🔗 Chaîne RAG créée.")

    # 4️⃣ Poser une question
    question = """
    Pour le compte de son nouveau client, Alliance Externe recherche un Développeur Full Stack Java/Angular confirmé (Statut consultant) pour accompagner l’un de nos clients dans la création de solutions digitales innovantes.
Le poste implique la conception et le développement d’applications web et mobiles d’entreprise, dans un environnement international et multiculturel.

Missions
• Développer des applications web et mobiles d’entreprise.
• Travailler en Agile Scrum au sein d’équipes internationales.
• Collaborer avec des designers pour créer des expériences utilisateurs de haut niveau.
• Participer aux revues de design et de code, assurer la qualité et les bonnes pratiques.
• Documenter et appliquer les bonnes pratiques (source control, CI/CD, issue tracking).
• Accompagner des développeurs juniors (mentorat) et apprendre des profils seniors.
• Intégrer et consommer des API tierces.
    """
    answer = db.ask(question)
    
    print("\n💬 Question :", question)
    print("✅ Réponse :", answer)
testRag()