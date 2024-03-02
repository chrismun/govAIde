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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

# Model configuration
model_path = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

# Helper functions
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
    # UI Configuration
    st.set_page_config(page_title="Support Inquiry", page_icon=":house_with_garden:")
    st.title("Support Services for Single Moms in Newark")
    
    support_option = st.selectbox("Select Support Category", ["Housing Support", "Childcare Support", "Education Support"])

    programs_dir = f'programs/{support_option.replace(" ", "_").lower()}'
    output_file = tempfile.mktemp(suffix='.txt')

    # Read and merge documents
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(programs_dir):
            filepath = os.path.join(programs_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as readfile:
                    outfile.write(readfile.read() + '\n')

    # Setup for document retrieval and question answering
    loader = TextLoader(file_path=output_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    persist_path = os.path.join(tempfile.gettempdir(), f"{support_option.replace(' ', '_').lower()}_vector_store.parquet")

    vector_store = SKLearnVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet",
    )

    question = st.text_input("How can we assist you today?", "")

    if st.button("Ask"):
        if question:
            ethical_check_prompt = f"Reviewing the ethics of the request: {question}"
            is_ethical = is_request_ethical(ethical_check_prompt)
            if not is_ethical:
                st.error("This request is not allowed, please try again.")
            else:
                with st.spinner('Searching for relevant information...'):
                    context = retrieve_document(question, vector_store)
                    if context == "No relevant document found.":
                        st.error(context)
                    else:
                        prompt = f'''
                        Question: {question}

                        Context: {context}

                        Response:
                        '''

                        answer = generate_text(prompt)
                        st.text_area("Response", answer, height=300)
        else:
            st.error("Please enter a question.")

    # Community Forum Feature
    if 'forum_posts' not in st.session_state:
        st.session_state.forum_posts = []

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

