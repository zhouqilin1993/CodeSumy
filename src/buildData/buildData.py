from pymongo import *
import json
import sys
import os

# build c# or java data from MongoDB
def build_data(lang):
    client = MongoClient("192.168.7.113",30000)
    db = client.code2text
    collection = db[lang+"_pairs"]

    dataset = []
    id = 0
    for doc in collection.find():
        id = id + 1
        title = doc["title"]
        code = doc["code"]
        docEntry = {"id": id, "title": title, "code":code }
        dataset.append(docEntry)

    f = open(os.environ["CodeSumy_HomeDir"] + '/data/' + lang + '/data_' + lang + ".json", 'w')
    f.write(json.dumps(dataset))
    f.close()

    return




if __name__ == '__main__':
    build_data("java")
    build_data("csharp")