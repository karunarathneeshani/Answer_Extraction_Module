from owlready2 import *
from rdflib.plugins.sparql import prepareQuery
from rdflib import URIRef

onto = get_ontology("file://E:/Academic/Final.owl").load()
graph = default_world.as_rdflib_graph()

UC = URIRef('http://www.semanticweb.org/hp/ontologies/2019/8/FinalProject#')
q = prepareQuery('''SELECT ?o
                       WHERE {
                                 ?subject UC:cantUpdate ?object;
 UC:answer ?o.}''', initNs={'UC': UC})
for row in graph.query(q):
    print(row)