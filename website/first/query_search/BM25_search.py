# from nltk.tokenize import word_tokenize


# sentence = 'Jack is a sharp minded fellow'
# words = word_tokenize(sentence)
# print(words)

# def spl_chars_removal(lst):
#     lst1=list()
#     for element in lst:
#         str=””
#         str = re.sub(“[⁰-9a-zA-Z]”,” “,element)
#         lst1.append(str)
#     return lst1


from rank_bm25 import BM25Okapi

corpus = [
    "Hello there windy good man!",
    "It is quite windy in London",
    "How is the weather windy today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

query = "windy London"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)
doc_scores = bm25.get_batch_scores(tokenized_query,[0,1,2])
print(doc_scores)
