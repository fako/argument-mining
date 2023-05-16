import json


class ChatGPTJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if hasattr("to_dict", obj):
            return obj.to_dict()
