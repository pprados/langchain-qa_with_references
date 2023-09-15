We believe that **hallucinations** pose a major problem in the adoption of LLMs (Language Model Models). 
It is imperative to provide a simple and quick solution that allows the user to verify the coherence of the answers 
to the questions they are asked. Don't forget that even answers to questions can be subject to hallucinations

The conventional approach is to provide a list of URLs of the documents that helped in answering (see qa_with_source). 
However, this approach is unsatisfactory in several scenarios:
1. The question is asked about a PDF of over 100 pages. Each fragment comes from the same document, but from where?
2. Some documents do not have URLs (data retrieved from a database or other *loaders*).

Other technical considerations make the URL approach tricky.
This is because prompts do not work well with complex URLs. This consumes a 
huge number of tokens, and in the end, the result is too big.

It appears essential to have a means of retrieving all references to the actual data sources 
used by the model to answer the question. 
It is better to return a list of `Documents` than a list of URLs.

This includes:
- The precise list of documents really used for the answer (the `Documents`, along with their metadata that may contain page numbers, 
slide numbers, or any other information allowing the retrieval of the fragment in the original document).
- The excerpts of text used for the answer in each fragment. Even if a fragment is used, the LLM only utilizes a 
small portion to generate the answer. Access to these verbatim excerpts helps to quickly ascertain the validity of the answer.

We propose a two pipelines: `qa_with_reference` and `qa_with_reference_and_verbatims` for this purpose. 
It is a Question/Answer type pipeline that returns the list of documents used, and in the metadata, the list of verbatim 
excerpts exploited to produce the answer. It is very similar to `qa_with_sources_chain` in that it is inspired by it.

If the verbatim is not really from the original document, it's removed.
# Install
```
pip install langchain-qa_with_reference
```

# Sample  notebook

See [here](https://github.com/pprados/langchain-qa_with_references/blob/master/qa_with_reference_and_verbatim.ipynb)

# langchain Pull-request
This is a temporary project while I wait for my langchain 
[pull-request](https://github.com/hwchase17/langchain/pull/5135) 
to be validated.

# It's experimental
For the moment, the code is being tested in a number of environments to validate and adjust it.
The langchain framework is very instable. Some features can become depreciated.
We work to maintain compatibility as far as possible.