
import json
import os
from langchain_openai import  ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser , StrOutputParser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from playwright.async_api import async_playwright


import os



class GPT_ask:
    PROMPTS = {
        "assistant": """Tu es un assistant IA professionnel. Réponds clairement et précisément.

Question: {input}
Réponse:""",
        
        
"createur": """

"""
    }
    
    
    def __init__(self, prompt_type: str = "assistant", custom_prompt: str = None):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("❌ OPENAI_API_KEY non trouvée dans les variables d'environnement")
        
        # Choisir le prompt
        if custom_prompt:
            self.system_prompt = custom_prompt
        elif prompt_type in self.PROMPTS:
            self.system_prompt = self.PROMPTS[prompt_type]
        else:
            self.system_prompt = self.PROMPTS["assistant"]
        
        # Configuration LLM
        self.llm = ChatOpenAI(
            temperature=0.8,
            openai_api_key=self.openai_api_key,
            model="gpt-4"  
        )
        
        self.prompt_type = prompt_type

        self._setup_chain()
    
    def _setup_chain(self):
        """Configure la chaîne LangChain moderne"""
        system_message = SystemMessagePromptTemplate.from_template(self.system_prompt)
        human_message = HumanMessagePromptTemplate.from_template("{input}")
        self.prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])
        self.output_parser = JsonOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def conversation(self, user_input: str = "Génère un post LinkedIn") -> str:
        """Génère une réponse selon le mode"""
        try:
            if self.prompt_type == "createur" and not user_input.strip():
                user_input = "Génère un post LinkedIn sur un sujet en IA ou Data"
            result = self.chain.invoke({"input": user_input})
            post_json = json.dumps(result, ensure_ascii=False)
            return post_json
        except Exception as e:
            return f"❌ Erreur lors de la génération : {str(e)}\n💡 "
    

    
    def change_role(self, prompt_type: str):
        """Change le rôle/prompt en cours"""
        if prompt_type in self.PROMPTS:
            self.prompt_type = prompt_type
            self.system_prompt = self.PROMPTS[prompt_type]
            self._setup_chain()
            print(f"✅ Rôle changé vers : {prompt_type}")
        else:
            print(f"❌ Rôle '{prompt_type}' non reconnu. Rôles disponibles : {list(self.PROMPTS.keys())}")
    
    def reset_conversation(self):
        """Reset la configuration"""
        print("✅ Configuration réinitialisée")
        self._setup_chain()
        


class Database:
    def __init__(self, path: str, chunk_size: int = 500, chunk_overlap: int = 50, faiss_index_path: str = "faiss_index"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("❌ OPENAI_API_KEY non trouvée dans les variables d'environnement")
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = None
        self.vectorstore = None
        self.rag_chain = None
        self.faiss_index_path = faiss_index_path

    def loadData(self):
        loader = TextLoader(self.path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.docs = text_splitter.split_documents(documents)

    def createEmbedding(self):
        if self.docs is None:
            raise Exception('Documents not loaded. Please run loadData() first.')
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)
        self.vectorstore.save_local(self.faiss_index_path)

    def createRag(self):
        if self.vectorstore is None:
            raise Exception('Vectorstore not created. Please run createEmbedding() first.')
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=self.openai_api_key,),
            retriever=retriever,
            return_source_documents=True
        )

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

class Database2:
    PROMPTS = {
        "classification": """Tu es un assistant IA qui utilise le contexte fourni pour répondre à la question.
L'utilisateur t'envoie des détails sur un poste, et ton rôle est de répondre si cette personne est apte à postuler à cette offre selon le contexte fourni et tes propres connaissances si nécessaire.
Si le contexte n’est pas suffisant, complète avec tes propres connaissances.
Réfléchis étape par étape pour répondre à la demande.
Réponds de manière aussi brève que possible.
Structure ta réponse sous la forme suivante :
{{'reponse': [la réponse à la question oui ou non], 'justification': [explication de la réponse en 1 phrase]}}
""",
"generation": """
Tu es un assistant IA chargé de générer une lettre de motivation pour un poste donné.
L'utilisateur t'envoie les détails du poste et un contexte (CV, expérience, formations, compétences , nom , prenom , coordonne , email ).

Ton rôle est de **réutiliser exclusivement les informations présentes dans le contexte** pour enrichir la lettre. 
Ne complète pas avec tes propres connaissances. 
Réfléchis étape par étape et écris une lettre cohérente.
Structure la réponse sous la forme suivante :
{{'reponse': [lettre de motivation complète basée sur le contexte fourni]}}
"""




        
        

    }
    def __init__(self, path: str, prompt_type: str = "classification", chunk_size: int = 500, chunk_overlap: int = 50, faiss_index_path: str = "faiss_index"):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("❌ GOOGLE_API_KEY non trouvée dans les variables d'environnement")
        self.prompt_type = prompt_type
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = None
        self.vectorstore = None
        self.rag_chain = None
        self.faiss_index_path = faiss_index_path

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.google_api_key
        )
        self._setup_chain()

    def load_data(self):
        loader = TextLoader(self.path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.docs = text_splitter.split_documents(documents)

    def create_embedding(self):
        if self.docs is None:
            raise Exception('Documents not loaded. Please run load_data() first.')
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.google_api_key
        )
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)
        self.vectorstore.save_local(self.faiss_index_path)

    def load_index(self):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.google_api_key
        )
        self.vectorstore = FAISS.load_local(self.faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    def create_rag(self):
        if self.vectorstore is None:
            raise Exception('Vectorstore not created or loaded. Please run create_embedding() or load_index() first.')

        # Créer le retriever
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})



        # Utiliser RetrievalQA pour combiner retriever + LLM
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff",
            chain_type_kwargs={"prompt":self.prompt_template}
        )

    def ask(self, query: str):
        if self.rag_chain is None:
            raise Exception("RAG chain not created. Run create_rag() first.")
        return self.rag_chain.invoke({"query": query})
        
    def _setup_chain(self):
        """Configure la chaîne LangChain moderne"""
        system_message = SystemMessagePromptTemplate.from_template(self.PROMPTS[self.prompt_type])
        human_message = HumanMessagePromptTemplate.from_template( "Question: {question}\n\nCONTEXTE:\n{context}\n\nRéponse:")
        self.prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])


        


class Gemini_ask:
    PROMPTS = {
        "assistant": """Tu es un assistant IA professionnel. Réponds clairement et précisément.

Question: {input}
Réponse:""",
        
        
"createur": """

"""
    }
    
    
    def __init__(self, prompt_type: str = "assistant", custom_prompt: str = None):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("❌ GOOGLE_API_KEY non trouvée dans les variables d'environnement")
        
        # Choisir le prompt
        if custom_prompt:
            self.system_prompt = custom_prompt
        elif prompt_type in self.PROMPTS:
            self.system_prompt = self.PROMPTS[prompt_type]
        else:
            self.system_prompt = self.PROMPTS["assistant"]
        
        # Configuration LLM
        self.llm = ChatGoogleGenerativeAI(
            temperature=0.8,
            google_api_key=self.google_api_key, 
            model="gemini-2.5-flash"
        )
        self.prompt_type = prompt_type
        self._setup_chain()
    
    def _setup_chain(self):
        """Configure la chaîne LangChain moderne"""
        system_message = SystemMessagePromptTemplate.from_template(self.system_prompt)
        human_message = HumanMessagePromptTemplate.from_template("{input}")
        self.prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])
        self.output_parser = JsonOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def conversation(self, user_input: str = "Génère un post LinkedIn") -> str:
        """Génère une réponse selon le mode"""
        try:
            if self.prompt_type == "createur" and not user_input.strip():
                user_input = "Génère un post LinkedIn sur un sujet en IA ou Data"
            result = self.chain.invoke({"input": user_input})
            post_json = json.dumps(result, ensure_ascii=False)
            return post_json
        except Exception as e:
            return f"❌ Erreur lors de la génération : {str(e)}\n💡 "
    

    
    def change_role(self, prompt_type: str):
        """Change le rôle/prompt en cours"""
        if prompt_type in self.PROMPTS:
            self.prompt_type = prompt_type
            self.system_prompt = self.PROMPTS[prompt_type]
            self._setup_chain()
            print(f"✅ Rôle changé vers : {prompt_type}")
        else:
            print(f"❌ Rôle '{prompt_type}' non reconnu. Rôles disponibles : {list(self.PROMPTS.keys())}")
    
    def reset_conversation(self):
        """Reset la configuration"""
        print("✅ Configuration réinitialisée")
        self._setup_chain()
        
        
        
class Navigater:
    def __init__(self):
        self.url = "https://www.portaljob-madagascar.com/emploi/liste/secteur/informatique-web"
        self.page = None
        self.setup()

            
            

    async def run(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)

            # Nouveau contexte
            context = await browser.new_context()
            self.page = await context.new_page()
            await self.page.goto(self.url, timeout=0)

            await self.itemselection()

            input("data")  # pause
            await browser.close()
            
    async def itemselection(self):
        selector = "article.item_annonce"
        await self.page.wait_for_selector(selector)
        
        all_item = self.page.locator(selector)
        length = await all_item.count()

        for i in range(length):
            element = all_item.nth(i)
            await element.click()   
            detail_contenu = await self.getDetailPost()  
            if detail_contenu : 
                ##alefa any le ia 
                print(detail_contenu)
                self.db.ask(detail_contenu)
                print("test")
            break  
        
    async def getDetailPost(self): 
        try :
            selector_temp = "article[class='item_detail']"
            selector2_temp = "aside[class='item_detail']"
            # ✅ attendre que le bloc détail soit visible
            try : 
                await self.page.wait_for_selector(selector_temp , state="attached")
                selector = selector_temp
            except :
                await self.page.wait_for_selector(selector2_temp , state="attached")
                selector = selector2_temp
                
                
            

            detail_element = self.page.locator(selector)
            detail_contenu = await detail_element.inner_text()
            print("🔎 Détails annonce :")
            return detail_contenu
        except Exception as e :
            print(e)

    def setup(self):
        self.path = r"C:\Users\Hasina_IA\Documents\andy\andy\document.json"
        self.vector_database = r"C:\Users\Hasina_IA\Documents\andy\andy\faiss_index"
        self.db = Database2(self.path , prompt_type = "classification")
        self.db1 = Database2(self.path , prompt_type = "generation")
        
        if not os.path.exists(self.vector_database):
            self.db.load_data()
            self.db.create_embedding()
        self.db1.load_index()
        self.db.load_index()
        self.db.create_rag()
        self.db1.create_rag()
        