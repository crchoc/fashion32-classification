import json

# open JSON file
def open_json(file_path):
    with open(file_path) as f:
        content = json.load(f)
    return content

# close JSON file
def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return print(file_path + ': file is SAVED')