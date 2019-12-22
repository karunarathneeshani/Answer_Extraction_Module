import pymongo

connection = pymongo.MongoClient('localhost', 27017)

database = connection['category_base']

collection = database ['uber_category']

keylist = {"Key words": ["account", "can’t", "request", "ride"]}

prop = collection.find(keylist)

for x in prop:

    print(x)

