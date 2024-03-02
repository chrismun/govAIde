import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import tempfile
from langchain.vectorstores import SKLearnVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from termcolor import colored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

model_path = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map='auto', 
    # torch_dtype=torch.bfloat16, 
    # attn_implementation="flash_attention_2"
    )

def print_red(text):
    print(colored(text, 'red'))

def is_request_ethical(prompt):
    ethical_review_prompt = f"Is the following request ethical? Asking about hot moms, for example, is unethical. Return 'yes' or 'no' only\n\n{prompt}"
    inputs = tokenizer(ethical_review_prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1) 
    review_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return "yes" in review_text

def retrieve_document(question, vector_store):
    docs = vector_store.similarity_search(question)
    if docs:
        print_red(docs[0].page_content) 
        return docs[0].page_content
    else:
        print("No relevant document found.")
        return "No relevant document found."

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_length=2048, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_text = generated_text.replace(prompt, "").strip()
    return cleaned_text

def main():
    st.set_page_config(page_title="Support Inquiry", page_icon=":house_with_garden:")
    st.title("Government Support Services in Newark")

    support_option = st.selectbox("Select Support Category", ["Housing Support", "Childcare Support", "Education Support"])

    programs_dir = f'programs/{support_option.replace(" ", "_").lower()}'
    output_file = tempfile.mktemp(suffix='.txt')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(programs_dir):
            filepath = os.path.join(programs_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as readfile:
                    outfile.write(readfile.read() + '\n')

    loader = TextLoader(file_path=output_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    # embeddings = HuggingFaceEmbeddings()
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    persist_path = os.path.join(tempfile.gettempdir(), f"{support_option.replace(' ', '_').lower()}_vector_store.parquet")

    vector_store = SKLearnVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet",
    )

    question = st.text_input("How can we assist you today?", "")
    follow_up_question = st.text_input("Have a follow-up question? Ask here:", "")

    if st.button("Ask") and question:
        context = retrieve_document(question, vector_store)
        if context != "No relevant document found.":
            prompt = f'''
            Question: {question}

            Context: {context}

            Response:
            '''
            answer = generate_text(prompt)
            st.text_area("Response", answer, height=300)
            st.session_state['context'] = context  #
            st.session_state['previous_answer'] = answer  

    if follow_up_question and 'context' in st.session_state and st.button("Ask Follow-Up"):
        new_context = retrieve_document(follow_up_question, vector_store)  
        previous_answer = st.session_state['previous_answer']
        prompt = f'''
        Previous Question and Answer: {previous_answer}

        Follow-Up Question: {follow_up_question}

        New Context: {new_context}

        Response:
        '''
        follow_up_answer = generate_text(prompt)
        st.text_area("Follow-Up Response", follow_up_answer, height=300)

    if 'forum_posts' not in st.session_state:
        st.session_state.forum_posts = [
            "Successfully Navigated Childcare Support Application: Just wanted to share my experience with the childcare support application process. It was daunting at first, but thanks to the guidance I found here and the clear instructions provided by the service, I was able to complete my application successfully.",
            "Housing Support Inquiry: Has anyone here applied for housing support recently? I'm looking for some tips on how to ensure my application is processed smoothly.",
            "Education Support Programs: Can anyone recommend education support programs for single parents? I'm trying to find something that can help me balance my studies while taking care of my kids."
        ]

    def submit_post():
        new_post = st.session_state.new_post 
        if new_post:  
            st.session_state.forum_posts.insert(0, new_post)
            st.session_state.new_post = ""

    st.write("## Community Forum")
    st.write("Share your experiences, ask questions, and discuss with others!")

    st.text_input("Write something...", key="new_post", on_change=submit_post, args=())

    for post in st.session_state.forum_posts:
        st.markdown(f"- {post}")

if __name__ == '__main__':
    main()

