
import json
class Outil : 
    @staticmethod
    def extract_from_json(value:str)->json :
        #result_json = value['result'].replace("'", '"')

        result_dict = json.loads(value)
        return result_dict