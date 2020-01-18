import pymongo
from collections import Counter
from owlready2 import *
from rdflib.plugins.sparql import prepareQuery
from rdflib import URIRef

connection = pymongo.MongoClient('localhost', 27017)
database = connection['category_base']
collection = database ['uber_category']

# a = ["account", "can’t", "forget", "password"]
a = ["account", "can’t", "request", "ride"]
squares = []

prop = collection.find({})
for documents in prop:
   b = documents['Key words']
   a_vals = Counter(a)
   b_vals = Counter(b)

   # convert to word-vectors
   words = list(a_vals.keys() | b_vals.keys())
   a_vect = [a_vals.get(word, 0) for word in words]  # [0, 0, 1, 1, 2, 1]
   b_vect = [b_vals.get(word, 0) for word in words]  # [1, 1, 1, 0, 1, 0]

   # find cosine
   len_a = sum(av * av for av in a_vect) ** 0.5  # sqrt(7)
   len_b = sum(bv * bv for bv in b_vect) ** 0.5  # sqrt(4)
   dot = sum(av * bv for av, bv in zip(a_vect, b_vect))  # 3
   cosine = dot / (len_a * len_b)
   if(cosine>0.6):
      squares.append(documents['Property'])
#print(squares)
if(len(squares)>0):
      onto = get_ontology("file://E:/Academic/Final.owl").load()
      graph = default_world.as_rdflib_graph()
      c = squares[0]
      UC = URIRef('http://www.semanticweb.org/hp/ontologies/2019/8/FinalProject#')
      q = prepareQuery('''SELECT ?o
                                 WHERE {
                                           ?subject UC:''' + c + ''' ?object;
           UC:answer ?o.}''', initNs={'UC': UC})

      results = graph.query(q)
      response = []
      for item in results:
         o = str(item['o'].toPython())
         o = re.sub(r'.*#', "", o)
         response.append(o)
         print(response)
else:
    onto = get_ontology("file://E:/Academic/Final.owl").load()
    graph = default_world.as_rdflib_graph()
    c = 'cantFind'
    UC = URIRef('http://www.semanticweb.org/hp/ontologies/2019/8/FinalProject#')
    q = prepareQuery('''SELECT ?o
                                 WHERE {
                                           ?subject UC:''' + c + ''' ?object;
           UC:answer ?o.}''', initNs={'UC': UC})

    results = graph.query(q)
    response = []
    for item in results:
        o = str(item['o'].toPython())
        o = re.sub(r'.*#', "", o)
        response.append(o)
        print(response)
