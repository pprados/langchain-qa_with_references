We believe that **hallucinations** pose a major problem in the adoption of LLMs (Language Model Models). 
It is imperative to provide a simple and quick solution that allows the user to verify the coherence of the answers 
to the questions they are asked.

The conventional approach is to provide a list of URLs of the documents that helped in answering (see qa_with_source). 
However, this approach is unsatisfactory in several scenarios:
1. The question is asked about a PDF of over 100 pages. Each fragment comes from the same document, but from where?
2. Some documents do not have URLs (data retrieved from a database or other *loaders*).

It appears essential to have a means of retrieving all references to the actual data sources used by the model to answer the question. 

This includes:
- The precise list of documents used for the answer (the `Documents`, along with their metadata that may contain page numbers, 
slide numbers, or any other information allowing the retrieval of the fragment in the original document).
- The excerpts of text used for the answer in each fragment. Even if a fragment is used, the LLM only utilizes a 
small portion to generate the answer. Access to these verbatim excerpts helps to quickly ascertain the validity of the answer.

We propose a two pipelines: `qa_with_reference` and `qa_with_reference_and_verbatims` for this purpose. 
It is a Question/Answer type pipeline that returns the list of documents used, and in the metadata, the list of verbatim 
excerpts exploited to produce the answer.

If the verbatim is not really from the original document, it's removed.
# Install
```
pip install langchain-qa_with_reference
```

# Sample  notebook

See [here]([https://github.com/pprados/langchain-qa_with_references/blob/master/qa_with_reference.ipynb](https://github.com/pprados/langchain-qa_with_references/blob/master/qa_with_reference_and_verbatim.ipynb)

# langchain Pull-request
This is a temporary project while I wait for my langchain 
[pull-request](https://github.com/hwchase17/langchain/pull/5135) 
to be validated.

