# retrieve answers from pubmed/galago
# open aueb pickle file and retrieve answers for each q
import sys
import pickle
import time
import requests
import json
import subprocess
import atexit
import unicodedata
import html
import multiprocessing
import pdb
from pubmed_lookup import PubMedLookup, Publication

# from tqdm import tqdm
import tqdm
import spacy

import os.path

import numpy as np

from sklearn.metrics import average_precision_score

from pubmed import get_doc_text
# mvieira  from galago import get_doc_text_galago

"""
Evaluate document retrieval systems on the corpora generated.
Currently implemented: galago and PubMed entrez API
Galago requries a local installation of pubmed
PubMed API requires API key stored in params.json
"""

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")


def process_search_results(ret_docs, aueb_dic, use_mp=True, get_doc_set=True, force_pmids=False):
    """Process document retrieval files to be used by AUEB system

    Update counts of each query, retrieve documents full text
    and generate bioasq json format

    :param ret_docs: docs retrieved for each query ID. {qid: {pmid: {rank, score, bm25}}}
    :type ret_docs: dict
    :param aueb_dic: AUEB format query dictionary ({queries:[{query_id, query_text, etc}]})
    """
    new_aueb_dic = {"queries": []}
    bioasqjson = {"questions": []}
    no_rel_ret_count = 0
    for r in aueb_dic["queries"]:
        # print(r)
        # print()
        # print('retrieved docs keys', ret_docs.keys())
        retrieved_docs = []
        qid = str(r["query_id"])
        if qid not in ret_docs:
            print("qid not found in ret_docs", qid, file=sys.stderr)
            # print(list(ret_docs.keys())[:10], qid, type(qid)) # 10 corresponds to K retrieved docs
            continue

        # helper list to add relevant_documents not retrieved by search_engine
        relevant_docs = r["relevant_documents"].copy()  # add-on

        # loop through the retrieved documents in ret_docs associated with this query
        for pmid in ret_docs[qid]:
            is_relevant = False
            if pmid in r["relevant_documents"]:
                is_relevant = True
                relevant_docs.remove(pmid)
            retrieved_docs.append(
                {
                    "doc_id": str(pmid),
                    "rank": ret_docs[qid][pmid]["rank"],
                    "bm25_score": ret_docs[qid][pmid].get(
                        "bm25", ret_docs[qid][pmid]["score"]
                    ),
                    "norm_bm25_score": ret_docs[qid][pmid]["bm25"],
                    "is_relevant": is_relevant,
                    # "score": ret_docs[qid][pmid]["score"],
                }
            )

        if force_pmids and len(relevant_docs) != 0:

            #  for remaining docs in 'relevant_documents' in biqa but not included in retrieved_docs
            #  create random high scores higher than the maximum scores of all pmids already in retrieved_docs

            # get highest bm_25_score and highest norm_bm25_score
            max_bm25 = []
            max_norm_bm25 = []

            relevant_docs_2 = []
            # check if all positive PMIDs HAVE ABSTRACT IN /pubmed_abstracts/
            for pmid in relevant_docs:
                #     if os.path.isfile('/pubmed_abstracts/{}.txt'.format(pmid)):
                relevant_docs_2.append(pmid)

            # decrease ranking of all docs in retrieved docs
            rank_count = len(relevant_docs_2)
            for docs in retrieved_docs:
                docs['rank'] = docs['rank'] + rank_count
                max_bm25.append(docs['bm25_score'])
                max_norm_bm25.append(docs['norm_bm25_score'])

            # because there might be cases where retrieved docs == 0
            # if len(retrieved_docs) == 0:
            #     max_bm25 = random.uniform(1, 5)
            #     max_norm_bm25 = random.uniform(1, 5)
            # else:
            max_bm25 = max(max_bm25)
            max_norm_bm25 = max(max_norm_bm25)

            beta_score = 1e-5
            new_rank = 1
            for pmid in relevant_docs_2:
                # random criterion: new  rank dependent on current order in the list.
                retrieved_docs.append(
                    {
                        "doc_id": str(pmid),
                        "rank": new_rank,
                        "bm25_score": max_bm25 + beta_score + rank_count,
                        "norm_bm25_score": max_norm_bm25 + beta_score + rank_count,
                        "is_relevant": True,
                        #                     "score": ret_docs[qid][pmid]["score"],
                    }
                )
                # add to ret_docs for doc set
                ret_docs[qid][pmid] = {
                    "rank": new_rank, 'score': max_bm25 + beta_score + rank_count}
                rank_count -= 1
                new_rank += 1

        # [print(i) for i in retrieved_docs]

        r["retrieved_documents"] = retrieved_docs
        r["num_ret"] = len(retrieved_docs)
        rel_ret = set(ret_docs[qid].keys()) & set(r["relevant_documents"])
        r["num_rel_ret"] = len(rel_ret)

        if len(rel_ret) == 0:
            no_rel_ret_count += 1

        new_aueb_dic["queries"].append(r)
        bioasq_query = {
            "body": r["query_text"],
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/" + d
                for d in r["relevant_documents"]
            ],
            "id": r["query_id"],
            "type": "",
            "snippets": [],
        }
        bioasqjson["questions"].append(bioasq_query)

    print('len of new_aueb_dic:', len(new_aueb_dic["queries"]))
    print(len(new_aueb_dic["queries"]) - no_rel_ret_count)

    if get_doc_set:
        docset = get_doc_set_info(ret_docs, new_aueb_dic, use_mp=use_mp)
    else:
        docset = None
    return new_aueb_dic, docset, bioasqjson


def get_doc_set_info(pmids_per_q, aueb_dic, use_mp=True):
    """ Return dic with pmid -> {doc_id: title, abstract}
    Either use a cache, or run with multiprocessing

    :param pmids_per_q: Dictionary with all question and respective PMIDs
    :type pmids_per_q: dict
    :return: Text of all PMIDs
    :rtype: dict

    """
    doc_set = {}
    all_pmids = []
    for q in pmids_per_q:
        all_pmids += pmids_per_q[q].keys()
    all_pmids = all_pmids
    print("retrieving doc text")
    if not use_mp:
        for pmid in tqdm(all_pmids):
            doc_object = get_doc_object(pmid)
            doc_set[str(pmid)] = doc_object
            # print("not using cache", pmid, type(pmid), doc_cache[int(pmid)])
    else:
        with multiprocessing.Pool(processes=20) as pool:
            # doc_objects = pool.map(get_doc_object, all_pmids)
            doc_objects = list(tqdm.tqdm(pool.imap(get_doc_object, all_pmids)))

            for i, doc in enumerate(doc_objects):
                doc_set[str(all_pmids[i])] = doc

    for pmid in all_pmids:
        if doc_set.get(str(pmid), None) is None:
            if str(pmid) in doc_set:
                del doc_set[str(pmid)]

    return doc_set


def get_doc_object(pmid):
    doc_info = get_doc_text(pmid)
    # doc_info = get_doc_text_galago(pmid)
    if doc_info is not None:
        doc_object = {
            "title": doc_info[0],
            "abstractText": doc_info[1],
            "publicationDate": "1950-01-01",
        }
        return doc_object
    else:
        # return None
        try:

            email = ''
            url = 'http://www.ncbi.nlm.nih.gov/pubmed/{}'.format(pmid)
            lookup = PubMedLookup(url, email)
            publication = Publication(lookup)
        except:
            return None
        else:

            if publication.abstract != '':
                doc_object = {
                    "title": str(publication.title),
                    "abstractText": str(publication.abstract),
                    #                 "abstractText": repr(publication.abstract),
                    "publicationDate": "1950-01-01",
                }
                print(pmid, 'abstract retrieved')
                return doc_object


def average_precision(ret_doc, rel_doc, max_items=10):
    """
    Calculate AP according to BioASQ guidelines.

    Instead of dividing by the number of relevant articles of each query, P is
    divided by the maximum number of documents

    :param ret_doc: ordered retrieved docs
    :type ret_doc: list
    :param rel_doc: relevant docs
    :type rel_doc: list
    :param max_items: max number of relevant items a query can have
    :type max_items: int

    """
    # order ret_doc by score
    # print(ret_doc, rel_doc)
    max_items = len(ret_doc)
    total = 0
    for i, doc in enumerate(ret_doc):
        if doc in rel_doc:
            tps = set(ret_doc[: i + 1]) & set(rel_doc)
            fps = set(ret_doc[: i + 1]) - set(rel_doc)
            partial = len(tps) / (len(tps) + len(fps))
            total += partial
    # print(total)
    return total / max_items


def calculate_scores(data, max_retrieve=10):
    """for each q-a pair, calculate micro p/r/f

    :param data: AUEB format dictionary
    :type data: dict
    :return: precision, recall, f1, map scores
    :rtype: tuple
    """
    new_data = {"queries": []}
    fps = 0
    tps = 0
    fns = 0
    maps = []
    for q in data["queries"]:
        # y_true = np.array([0, 0, 1, 1])
        y_true = []
        y_scores = []
        ret_scores = {}
        for retdoc in q["retrieved_documents"]:
            # print(retdoc)
            # ret_scores[retdoc["doc_id"]] = retdoc["score"]
            ret_scores[retdoc["doc_id"]] = retdoc["bm25_score"]
            if retdoc["doc_id"] in q["relevant_documents"]:
                tps += 1
                y_true.append(1)
                # y_scores.append(retdoc["score"])
                y_scores.append(retdoc["bm25_score"])
            else:
                fps += 1
                y_true.append(1e-10)
                # y_scores.append(retdoc["score"])
                y_scores.append(retdoc["bm25_score"])
        for reldoc in set(q["relevant_documents"]):
            # print(reldoc)fprint
            if reldoc not in [x["doc_id"] for x in q["retrieved_documents"]]:
                fns += 1
        if tps > 0:
            new_data["queries"].append(q)
        # print(y_true, y_scores)
        try:
            doc_ap = average_precision_score(y_true, y_scores)
        # doc_ap = average_precision(
        #    [k for k, v in sorted(ret_scores.items(), key=lambda item: item[1])],
        #    q["relevant_documents"],
        # )
        except:
            doc_ap = 0.0
        if np.isnan(doc_ap):
            maps.append(0.0)
        else:
            maps.append(doc_ap)
        # if doc_ap > 0.3:
        #    print(q["query_id"], doc_ap, set(q["relevant_documents"]))

    if tps + fps > 0:
        p = tps / (tps + fps)
    else:
        p = 0

    if tps + fns > 0:
        r = tps / (tps + fns)
    else:
        r = 0

    if p > 0 and r > 0:
        f = (2 * p * r) / (p + r)
    else:
        f = 0
    # print(maps)
    map_score = sum(maps) / len(maps)
    # gmap_score = p.exp(np.mean(np.log(np.array(maps))))
    # a = np.log(maps)
    # gmap_score = np.exp(a.sum() / len(a))
    print("TPs:", tps, len(data["queries"]))
    return (p, r, f, map_score), new_data  # , gmap_score


def main():
    # 1: retrieval engine
    retrieval_engine = sys.argv[1]

    # 1: AUEB pickle file
    with open(sys.argv[2], "rb") as f:
        data = pickle.load(f)
    print('queries:', len(data["queries"]))
    # choose topk documents from query results
    topk = 100
    force_pmids = True

    get_doc_set = True
    use_mp = True
    print('topk', topk)
    print('force_pmids', force_pmids)
    limit_queries = None
    # max number of queries to perform (ignore the other qs)
    # limit_queries = 100
    # could be either number of list
    # limit_queries = ["3448"]

    # use galago or pubmed to retrieve documents

    # data['queries'] = data['queries'][:]
    if retrieval_engine.startswith("galago"):

        from galago import get_pmids_galago
        if retrieval_engine.endswith("bm25"):
            galago_ret_docs = get_pmids_galago(
                data, n=topk, bm25=True, limit_queries=limit_queries)
        else:
            galago_ret_docs = get_pmids_galago(
                data, n=topk, bm25=False, limit_queries=limit_queries)

        data, docset, bioasqjson = process_search_results(
            galago_ret_docs, data, get_doc_set, use_mp, force_pmids=force_pmids)

    elif retrieval_engine == "pubmed":  # use pubmed
        from pubmed import get_pubmeds_for_questions

        pubmed_ret_docs = get_pubmeds_for_questions(
            data, n_docs=topk, limit_queries=limit_queries
        )
        data, docset, bioasqjson = process_search_results(
            pubmed_ret_docs, data,  use_mp=True, force_pmids=force_pmids)

    elif retrieval_engine == "drqa":
        from drqa_retriever import get_pmids_drqa

        drqa_ret_docs = get_pmids_drqa(
            data, n=topk, limit_queries=limit_queries)
        data, docset, bioasqjson = process_search_results(
            drqa_ret_docs, data, get_doc_set, use_mp, force_pmids=force_pmids)
    elif retrieval_engine == "elasticsearch":
        import esearch

        esearch_ret_docs = esearch.get_pubmeds_for_questions(
            data, n=topk, limit_queries=limit_queries
        )
        data, docset, bioasqjson = process_search_results(
            esearch_ret_docs, data, get_doc_set, use_mp, force_pmids=force_pmids)
    # print(data)
    scores, data = calculate_scores(data, topk)
    print(sys.argv[1:], scores)

    # write_data = False

    if len(sys.argv) > 3:

        bm25_data_path_train = os.path.join(
            sys.argv[3] + ".top{0}.pacrr.pkl".format(topk))
        docset_path_train = os.path.join(
            sys.argv[3] + ".docset_top{0}.pacrr.pkl".format(topk)
        )

        # check if all ret PMIDs are in docset
        if get_doc_set:
            for q in data["queries"]:
                for doc in q["retrieved_documents"]:
                    if str(doc["doc_id"]) not in docset:
                        print("doc id not in docset!", doc["doc_id"])
                        print(list(docset.keys()))  # [:10])

        with open(bm25_data_path_train, "wb") as f:
            data_train = pickle.dump(data, f)

        with open(docset_path_train, "wb") as f:
            docset_train = pickle.dump(docset, f)

        with open(sys.argv[3] + ".top{}.qrel.json".format(topk), "w") as f:
            json.dump(bioasqjson, f)

        with open(sys.argv[1] + ".pubmed", "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
