
from langgraph.graph import StateGraph, END

from Model.GraphState import GraphState

from outil.outil import Outil
##~classification et generation 
class Graph : 
    def __init__(self , llm_classification_instance, llm_generation_instance  ):
        print("✅ Générateur initialisé avec succès !")
        self.classification_model  = llm_classification_instance
        self.generation_model = llm_generation_instance
        self.app = self.create_processus_graph()
    
    def create_processus_graph(self):
        self.graph = StateGraph(GraphState)
        self.graph.add_node("classification", self.classification)
        self.graph.add_node("generation", self.generation)
        

        self.graph.set_entry_point("classification")
        
        self.graph.add_conditional_edges(
            "classification", 
            lambda x: x.get("next", "END"),
            {"generation":"generation","END":END}
        )
        self.graph.add_conditional_edges(
            "generation", 
            lambda x: x.get("next", "END"),
            
        )
        return self.graph.compile()
    
    def getApp(self):
        if not self.app :
            raise Exception("class need to be instancified")
        return self.app
    
    def classification(self , state):
        try:                        
            content : str | None = state.get("post_info")
            if  not  content :
                return "END"
            post = self.classification_model.ask(content)
            print(f"post : {post}")
            print("*"*50)
            result = post["result"]
            response_value = Outil.extract_from_json(result)
            print(response_value)
            reponse = response_value.get("reponse")
            print(reponse)
            
            if  reponse == None:
                raise Exception("reponse absent")
            try :
                response_int = int(reponse)
                if response_int == 1 : 
                    state["next"] = "generation"
                else : 
                    state["next"] = "END"
            except Exception : 
                    state["next"] = "END"
            return state
        except Exception as e:
            print(f"❌ Erreur : {e}")
            state['next'] = "END"
            return state

    def generation(self,state):
        print("📱 GÉNÉRATION D'UN LM AUTOMATIQUE :")
        try :
            content : str | None = state.get("post_info")
            if  not  content :
                return "END"
            lm_data = self.generation_model.ask(content)
            result = lm_data["result"]
            response_value = Outil.extract_from_json(result)
            
            lm_value = response_value.get("reponse")
            if  lm_value == None:
                raise Exception("lm absent")
            state["lm"] = lm_data
        except Exception as e : 
            print(f"erreur : {str(e)} ")
        state["next"] = "END"
        return state


