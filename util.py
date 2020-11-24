import json

def pretty_print_json(item):
    print(json.dumps(item, indent=4, sort_keys=True))