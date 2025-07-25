from load_docs import load_or_create_vectorstore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def create_qa_chain():
    retriever = load_or_create_vectorstore("docs")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def main():
    qa_chain = create_qa_chain()
    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
        answer = qa_chain.invoke(query)
        print("\n\U0001F9E0 Answer:\n" + answer['result'])

if __name__ == "__main__":
    main()