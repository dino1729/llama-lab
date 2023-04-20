import os
from llama_index import (
    GPTSimpleVectorIndex,
    GPTListIndex,
    ServiceContext,
    LLMPredictor,
)

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
#llm = AzureOpenAI(temperature=0, deployment_name="text-davinci-003", max_tokens=512)
llm = AzureChatOpenAI(temperature=0, deployment_name="gpt-3p5-turbo", max_tokens=512)
llm_predictor = LLMPredictor(llm=llm)


def initialize_task_list_index(
    documents, llm_predictor=llm_predictor, index_path="./task_index.json", chunk_size_limit=2000
):
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=chunk_size_limit
    )
    if os.path.exists(index_path):
        return GPTListIndex.load_from_disk(index_path, service_context=service_context)
    else:
        return GPTListIndex.from_documents(documents, service_context=service_context)


def initialize_search_index(
    documents,
    llm_predictor=llm_predictor,
    index_path="./search_index.json",
    chunk_size_limit=2000,
):
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=chunk_size_limit
    )
    if os.path.exists(index_path):
        return GPTSimpleVectorIndex.load_from_disk(
            index_path, service_context=service_context
        )
    else:
        return GPTSimpleVectorIndex.from_documents(
            documents, service_context=service_context
        )


def log_current_status(cur_task, result, completed_tasks_summary, task_list):
    status_string = f"""
    ==================================
    Completed Tasks Summary: {completed_tasks_summary.strip()}
    Current Task: {cur_task.strip()}
    Result: {result.strip()}
    Task List: {", ".join([x.get_text().strip() for x in task_list])}
    ==================================
    """
    print(status_string, flush=True)
