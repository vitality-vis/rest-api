import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "VitaLITy-2.0.0.json"


def check(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)

    ada_null_list = []
    specter_null_list = []
    glove_null_list = []

    for i in data:
        if not i.get('ada_embedding'):
            ada_null_list.append(i['ID'])
        if not i.get('specter_embedding'):
            specter_null_list.append(i['ID'])
        if not i.get('glove_embedding'):
            glove_null_list.append(i['ID'])

    print(ada_null_list)
    print(specter_null_list)
    print(glove_null_list)


if __name__ == '__main__':
    json_file_path = DEFAULT_JSON_PATH
    check(json_file_path)
