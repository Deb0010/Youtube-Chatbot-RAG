# %%


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

st.title("Youtube Chatbot with RAG")

video_id = st.text_input("Give Video ID")
# %%
# video_id = "Gfr50f6ZBvo" # only the ID, not full URL
if video_id:
    try:
        # If you don’t care which language, this returns the “best” one
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

        # Flatten it to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

    except TranscriptsDisabled:
        print("No captions available for this video.")

    # %%
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # %%
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    import os
    os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)

    # %%
    retriever=db.as_retriever()

    # %%
    from google import genai
    from langchain.chat_models import init_chat_model
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    # %%
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )

    # %%
    question          = "is the topic of nuclear fusion discussed in this video?"
    retrieved_docs    = retriever.invoke(question)

    # # %%
    # retrieved_docs

    # %%
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # %%
    final_prompt = prompt.invoke({"context": context_text, "question": question})

    # %%
    # final_prompt

    # %%
    answer = model.invoke(final_prompt)
    print(answer.content)

    # %%
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    # %%
    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    # %%
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })


    # %%
    parser = StrOutputParser()

    # %%
    main_chain = parallel_chain | prompt | model| parser

    # %%
    # response = main_chain.invoke('Can you summarize the video in 1 line')
    # print(response)


    topic = st.text_input("What question do you want to ask about the video?")
    if topic:
        result = main_chain.invoke(topic)
        st.write(result)