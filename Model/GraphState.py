from typing import TypedDict

class GraphState(TypedDict, total=False):
    # Input data from email
    post_info : str 
    lm: str
    next : str
    
    