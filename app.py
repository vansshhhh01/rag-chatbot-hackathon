import streamlit as st
import os
import torch
import re
import fitz  # PyMuPDF library
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# --- Core Functions (No changes here) ---

def get_pdf_text(pdf_docs):
    all_text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for page in doc:
                all_text += page.get_text()
    return all_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_llm():
    hf_token = "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" 
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map={'': 0}, token=hf_token)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, return_full_text=False)
    return HuggingFacePipeline(pipeline=pipe)

def get_synthesizer_chain(llm):
    template = "[INST] You are an expert report writer... Original Research Topic: {topic}\nCollected Research (Questions and Answers):\n{research_data}\n[/INST]\nFinal Summary Report:"
    prompt = PromptTemplate(template=template, input_variables=["topic", "research_data"])
    return LLMChain(prompt=prompt, llm=llm)

def get_planner_chain(llm):
    template = "[INST] You are an expert research analyst... Topic: {topic} [/INST]\nQuestions:"
    prompt = PromptTemplate(template=template, input_variables=["topic"])
    return LLMChain(prompt=prompt, llm=llm)

def get_conversation_chain(vectorstore, text_chunks, llm):
    template = "[INST] You are a helpful AI assistant... Here is the context:\n{context}\nHere is the user's question:\n{question} [/INST]"
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    bm25_retriever = BM25Retriever.from_texts(text_chunks)
    bm25_retriever.k = 5
    faiss_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=ensemble_retriever, return_source_documents=True, combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return chain

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="AI Research Analyst 📈", layout="wide")
    st.header("AI Research Analyst 📈")

    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("processed_text_chunks", [])
    st.session_state.setdefault("planner_chain", None)
    st.session_state.setdefault("synthesizer_chain", None)
    st.session_state.setdefault("research_plan", None)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.processed_text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(st.session_state.processed_text_chunks)
                    llm = get_llm()
                    st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.processed_text_chunks, llm)
                    st.session_state.planner_chain = get_planner_chain(llm)
                    st.session_state.synthesizer_chain = get_synthesizer_chain(llm)
                    
                    # --- UPGRADED VISUAL CONFIRMATION (FINDS ALL IMAGES) ---
                    st.sidebar.subheader("Images Found")
                    images_found = 0
                    for pdf in pdf_docs:
                        pdf.seek(0)
                        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
                            for page_num in range(len(doc)):
                                page = doc.load_page(page_num)
                                for img_index, img in enumerate(page.get_images(full=True)):
                                    xref = img[0]
                                    base_image = doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    st.sidebar.image(
                                        image_bytes, 
                                        caption=f"Image {images_found+1} (Page {page_num+1})",
                                        use_column_width=True
                                    )
                                    images_found += 1
                    if images_found == 0:
                        st.sidebar.write("No images were found in the document(s).")
                    else:
                        st.sidebar.success(f"Found {images_found} total image(s)!")
                    # --- END OF UPGRADED CODE ---

                    st.success("Ready to Chat or Analyze!")
                    st.session_state.chat_history = []
                    st.session_state.research_plan = None
            else:
                st.warning("Please upload at least one PDF file.")

    # (The rest of the main function with the Chat and Analysis tabs remains unchanged)
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Autonomous Analysis"])
    with tab1:
        st.subheader("Direct Q&A")
        # ... (rest of chat logic)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Show Sources"):
                        for i, doc in enumerate(message["sources"]):
                            st.info(f"Source {i+1}")
                            st.write(doc.page_content)
        if user_question := st.chat_input("Ask a question about your documents..."):
            if st.session_state.conversation:
                # ...
                with st.chat_message("user"):
                    st.markdown(user_question)
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    simple_history = [(msg["content"], st.session_state.chat_history[i+1]["content"]) for i, msg in enumerate(st.session_state.chat_history[:-1]) if msg["role"] == "user"]
                    response_stream = st.session_state.conversation.stream({"question": user_question, "chat_history": simple_history})
                    sources = []
                    for chunk in response_stream:
                        if "answer" in chunk:
                            full_response += chunk["answer"]
                            message_placeholder.markdown(full_response + "▌")
                        if "source_documents" in chunk:
                            sources = chunk["source_documents"]
                    message_placeholder.markdown(full_response)
                    if sources:
                        with st.expander("Show Sources"):
                            for i, doc in enumerate(sources):
                                st.info(f"Source {i+1}")
                                st.write(doc.page_content)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "sources": sources})
            else:
                st.warning("Please process your documents first.")

    with tab2:
        # ... (rest of analysis logic)
        st.subheader("Generate an Automated Report")
        research_topic = st.text_input("Enter a research topic (e.g., 'Analyze the marketing strategy')", key="research_topic_input")
        if st.button("Generate Plan"):
            if st.session_state.planner_chain and research_topic:
                with st.spinner("The AI is thinking of a research plan..."):
                    plan_response = st.session_state.planner_chain.invoke({"topic": research_topic})
                    st.session_state.research_plan = plan_response['text']
            else:
                st.warning("Please process documents and enter a topic first.")
        if st.session_state.research_plan:
            st.info("AI-Generated Research Plan:")
            st.markdown(st.session_state.research_plan)
            if st.button("Generate Full Report from this Plan"):
                research_data = []
                with st.status("Executing research plan...", expanded=True) as status:
                    questions = [q.strip() for q in re.findall(r'^\d+\.\s*(.*)', st.session_state.research_plan, re.MULTILINE)]
                    for i, question in enumerate(questions):
                        st.write(f"Answering question {i+1}/{len(questions)}: *{question}*")
                        response = st.session_state.conversation({"question": question, "chat_history": []})
                        research_data.append(f"Question: {question}\nAnswer: {response['answer']}")
                    status.update(label="Synthesizing final report...", state="running")
                    research_data_str = "\n\n---\n\n".join(research_data)
                    final_report = st.session_state.synthesizer_chain.invoke({"topic": st.session_state.research_topic_input, "research_data": research_data_str})
                    status.update(label="Report Generation Complete!", state="complete", expanded=False)
                st.subheader("Final Synthesized Report")
                st.markdown(final_report['text'])
                st.session_state.research_plan = None
if __name__ == '__main__':
    main()