# pubmed api interface
import json
import os
import time
import html
import requests
import spacy

from tqdm import tqdm



def get_doc_text(pmid, abstract_path="/pubmed_abstracts/"):
    """Retrieve text from PubMed files stored on disk

    :param pmid: PubMed ID to retrieve
    :type pmid: string
    :param abstract_path: directory where pubmed text files are stored as .txt files
    :type abstract_path: string
    :return: Title and abstract of article
    :rtype: string

    """
    if "http" in pmid:  # extract pmid
        pmid = pmid.split("/")[-1]

    if not os.path.isfile(abstract_path + pmid + ".txt"):
        # either the pmid is wrong or pubmed doesnt have an abstract in text format
        # print(pmid, 'text not found -  in get_doc_text from pubmed.py')
        return None
        # return ("", "")

    with open(abstract_path + pmid + ".txt") as f:
        text = f.readlines()
    # print(text)

    if text[0].strip() == "":  # and text[-1].strip() != "":

        # print("no text", text)

        # return None
        return ('no text', " ".join(text[1:]).strip())

    # elif text[0].strip() != "" and text[-1].strip() == "":
    #     return (text[0].strip(), 'no text')

    return (text[0].strip(), " ".join(text[1:]).strip())


def get_pmids_for_query(query, n_docs, n_tokens=20, n_chars=500):
    """ Use PubMed entrez api to retrieve documents according to a query

    Query processing is performed on this function as it might differ from other
    retrieval engines.
    The system waits 0.1 seconds between each request to prevent from going over the
    10 requests per second limit.

    :param query: Natural language query
    :type query: string
    :param n_docs: max number of documents to retrieve
    :type n_docs: int
    :param n_tokens: max number of token of the query
    :type n_tokens: int
    :param n_chars: max number of chars of the query (including URL)
    :type n_chars: int
    :return: list of PMIDs
    :rtype: list


    """
    # field=tiab&
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key={}&db=pubmed&retmode=json&sort=relevance&retmax={}&term={}"
    query = html.unescape(query)
    doc = nlp(query)
    doc_tokens = [
        t for t in doc if not t.is_punct and not t.is_space and not t.is_stop]

    doc_tokens = sorted(doc_tokens, key=lambda x: x.prob, reverse=False)

    doc_tokens = list(dict.fromkeys([t.text.lower() for t in doc_tokens]))
    doc_tokens = doc_tokens[:n_tokens]

    request_url = base_url.format(
        params["pubmed_api"], n_docs, "+OR+".join(doc_tokens))
    if len(request_url) > n_chars:
        print("long url! trimming to {}".format(n_chars))
        request_url = request_url[:n_chars]
    try:
        pubmed_results = requests.get(request_url)
    except:
        return []
    # print(request_url, pubmed_results.text)
    if pubmed_results.status_code != 200:
        print(pubmed_results.text)

    if "json" not in pubmed_results.headers.get("Content-Type"):
        print("Response content is not in JSON format.")
        print(pubmed_results.text)
        return []
    try:
        pubmed_results = pubmed_results.json()
    except json.decoder.JSONDecodeError:
        return []

    try:
        pmids = pubmed_results["esearchresult"]["idlist"]
    except KeyError:
        print("KEYERROR no IDs")
        pmids = []
    # print(request_url, len(pmids))
    time.sleep(0.1)
    return pmids


def get_pubmeds_for_questions(query_text, n_docs=100):

    pmids = get_pmids_for_query(query_text, n_docs)
    print(pmids)
    # nresults_count.append(len(pmids))
    # for i, pmid in enumerate(pmids):
    #     ret_docs[query_text][pmid] = {"rank": i,
    #                             "score": (len(pmids) - i) / len(pmids)}
    # print("average n of results", sum(nresults_count) / len(nresults_count))
    # return ret_docs



# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")

with open("params.json", "r") as f:
    params = json.load(f)

query_text='How many coronary arteries exist?'

pubmed_ret_docs = get_pubmeds_for_questions(query_text, n_docs=100)
# data, docset, bioasqjson = process_search_results(pubmed_ret_docs, data,  use_mp=True, force_pmids=force_pmids)