from pycorenlp import StanfordCoreNLP

coreNLPInstance = StanfordCoreNLP('http://localhost:9000')
print('[INFO] Connected to Running Instance')
sentence = input('Enter Sentence: ')
output = coreNLPInstance.annotate(sentence, properties={'annotators': 'openie', 'outputFormat':'json'})
print('[INFO] Possible Knowledge triples')
openIeObject = output['sentences'][0]['openie']
for rel in openIeObject:
    subject = rel['subject']
    relation = rel['relation']
    obj = rel['object']
    knowledgeTripletTuple =(subject, relation, obj)
    print(knowledgeTripletTuple)
print('[INFO] Knowledge triplets extracted')