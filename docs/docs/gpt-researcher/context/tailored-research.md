# Tailored Research

The GPT Researcher package allows you to tailor the research to your needs such as researching on specific sources (URLs) or local documents, and even specify the agent prompt instruction upon which the research is conducted.

### Research on Specific Sources 📚

You can specify the sources you want the GPT Researcher to research on by providing a list of URLs. The GPT Researcher will then conduct research on the provided sources via `source_urls`. 

If you want GPT Researcher to perform additional research outside of the URLs you provided, i.e., conduct research on various other websites that it finds suitable for the query/sub-query, you can set the parameter `complement_source_urls` as `True`. Default value of `False` will only scour the websites you provide via `source_urls`.


```python
from gpt_researcher import GPTResearcher
import asyncio

async def get_report(query: str, report_type: str, sources: list) -> str:
    researcher = GPTResearcher(query=query, report_type=report_type, source_urls=sources, complement_source_urls=False)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report

if __name__ == "__main__":
    query = "What are the biggest trends in AI lately?"
    report_source = "static"
    sources = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.ibm.com/think/insights/artificial-intelligence-trends",
        "https://www.forbes.com/advisor/business/ai-statistics"
    ]
    report = asyncio.run(get_report(query=query, report_source=report_source, sources=sources))
    print(report)
```

### Specify Agent Prompt 📝

You can specify the agent prompt instruction upon which the research is conducted. This allows you to guide the research in a specific direction and tailor the report layout.
Simply pass the prompt as the `query` argument to the `GPTResearcher` class and the "custom_report" `report_type`.

```python
from gpt_researcher import GPTResearcher
import asyncio

async def get_report(prompt: str, report_type: str) -> str:
    researcher = GPTResearcher(query=prompt, report_type=report_type)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report
    
if __name__ == "__main__":
    report_type = "custom_report"
    prompt = "Research the latest advancements in AI and provide a detailed report in APA format including sources."

    report = asyncio.run(get_report(prompt=prompt, report_type=report_type))
    print(report)
```

### Research on Local Documents 📄
You can instruct the GPT Researcher to research on local documents by providing the path to those documents. Currently supported file formats are: PDF, plain text, CSV, Excel, Markdown, PowerPoint, and Word documents.

*Step 1*: Add the env variable `DOC_PATH` pointing to the folder where your documents are located.

For example:

```bash
export DOC_PATH="./my-docs"
```

*Step 2*: When you create an instance of the `GPTResearcher` class, pass the `report_source` argument as `"local"`.

GPT Researcher will then conduct research on the provided documents.

```python
from gpt_researcher import GPTResearcher
import asyncio

async def get_report(query: str, report_source: str) -> str:
    researcher = GPTResearcher(query=query, report_source=report_source)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report
    
if __name__ == "__main__":
    query = "What can you tell me about myself based on my documents?"
    report_source = "local" # "local" or "web"

    report = asyncio.run(get_report(query=query, report_source=report_source))
    print(report)
```

### Hybrid Research 🔄
You can combine the above methods to conduct hybrid research. For example, you can instruct the GPT Researcher to research on both web sources and local documents.
Simply provide the sources and set the `report_source` argument as `"hybrid"` and watch the magic happen.

Please note! You should set the proper retrievers for the web sources and doc path for local documents for this to work.
To learn more about retrievers check out the [Retrievers](https://docs.gptr.dev/docs/gpt-researcher/search-engines/retrievers) documentation.


### Research on LangChain Documents 🦜️🔗
You can instruct the GPT Researcher to research on a list of langchain document instances.

For example:

```python
from langchain_core.documents import Document
from typing import List, Dict
from gpt_researcher import GPTResearcher
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine
import asyncio



CONNECTION_STRING = 'postgresql://someuser:somepass@localhost:5432/somedatabase'

def get_retriever(collection_name: str, search_kwargs: Dict[str, str]):
    engine = create_engine(CONNECTION_STRING)
    embeddings =  OpenAIEmbeddings()

    index = PGVector.from_existing_index(
        use_jsonb=True,
        embedding=embeddings,
        collection_name=collection_name,
        connection=engine,
    )

    return index.as_retriever(search_kwargs=search_kwargs)


async def get_report(query: str, report_type: str, report_source: str, documents: List[Document]) -> str:
    researcher = GPTResearcher(query=query, report_type=report_type, report_source=report_source, documents=documents)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report

if __name__ == "__main__":
    query = "What can you tell me about blue cheese based on my documents?"
    report_type = "research_report"
    report_source = "langchain_documents"

    # using a LangChain retriever to get all the documents regarding cheese
    # https://api.python.langchain.com/en/latest/retrievers/langchain_core.retrievers.BaseRetriever.html#langchain_core.retrievers.BaseRetriever.invoke
    langchain_retriever = get_retriever("cheese_collection", { "k": 3 })
    documents = langchain_retriever.invoke("All the documents about cheese")
    report = asyncio.run(get_report(query=query, report_type=report_type, report_source=report_source, documents=documents))
    print(report)
```
