#hi, who are you?

#--------ignore this-------------------------------------------
import os
# Kill TensorFlow logs BEFORE it initializes
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Now import everything else
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

def build_rag(yt_url):
    loader = YoutubeLoader.from_youtube_url(
        yt_url,
        add_video_info=False
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embedder)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return retriever
def engine(retriever, query):

    def format_context(results):
        return "\n\n".join([doc.page_content for doc in results])

    llm = HuggingFaceEndpoint(
        task="text-generation",
        model="google/gemma-2-2b-it"
    )
    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.
Make the answer detailed.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )
    chain = (
        RunnableParallel({
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(format_context)
        })
        | prompt
        | model
        | parser
    )
    response = chain.invoke(query)
    print("\nResponse:\n", response)
    
def main():
    print("Ask only about the video content. One question at a time.\n")
    while True:
        url = input("Enter YouTube video URL: ").strip()
        retriever = build_rag(url)

        while True:
            question = input("Enter your question: ").strip()
            engine(retriever, question)

            if input("Another question on SAME video? (y/n): ").strip().lower() != 'y':
                break

        if input("Ask about ANOTHER video? (y/n): ").strip().lower() != 'y':
            print("\nTHIS PROJECT WAS DONE BY SAHIL RANAKOTI... THANK YOU FOR USING IT!")
            break

if __name__ == "__main__":
    main()