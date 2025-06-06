# coding: utf-8
import json
import os.path

from pymongo import MongoClient

import config


# This onetime script will load json to mongodb
def load_json_to_mongo():
    client = MongoClient(config.mongodb_connection_uri)
    db = client[config.mongodb_database]
    collection = db[config.mongodb_docs_collection]

    json_file_path = os.path.join(config.PROJ_ROOT_DIR, config.raw_json_datafile)

    with open(json_file_path) as f:
        data = json.load(f)
        for i in data:
            collection.insert_one(i)
