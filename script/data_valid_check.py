import json


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
    # json_file_path = os.path.join(config.PROJ_ROOT_DIR, config.raw_json_datafile)
    json_file_path = '../data/VitaLITy-2.0.0.json'
    check(json_file_path)
