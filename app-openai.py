import streamlit as st
import os
import tempfile
from langchain.vectorstores import SKLearnVectorStore
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from termcolor import colored
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


# os.environ["OPENAI_API_KEY"] = 'key'

client=OpenAI()


def print_red(text):
    """Print text in red color."""
    print(colored(text, 'red'))

# def is_request_ethical(prompt):
#     """Check if a request is ethical."""
#     ethical_review_prompt = f"Is the following request ethical? Asking about hot moms, for example, is unethical. Return 'yes' or 'no' only\n\n{prompt}"
#     inputs = tokenizer(ethical_review_prompt, return_tensors="pt", max_length=500, truncation=True).to(device)
#     outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
#     review_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
#     return "yes" in review_text

def retrieve_document(question, vector_store):
    """Perform RAG"""
    docs = vector_store.similarity_search(question)
    if docs:
        print_red(docs[0].page_content) 
        return docs[0].page_content + docs[1].page_content
    else:
        print("No relevant document found.")
        return "No relevant document found."

def generate_text(prompt):
    """Generate text based on a prompt using OpenAI GPT-3.5 Turbo."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
                    {"role": "system", "content": "You are a helpful assistant for delaware government aid programs and job searches.."},
                    {"role": "user", "content": prompt}
                ]
    )
    return response.choices[0].message.content

def main_page():
    """Main Support Services Page."""

    # Landing page 
    st.title("govAIde")
    st.subheader("Government Support Services in Delaware")
    
    support_option = st.selectbox("Select Support Category", ["Housing Support", "Childcare Support", "Education Support", "Job Search"])
    programs_dir = f'programs/{support_option.replace(" ", "_").lower()}'
    output_file = tempfile.mktemp(suffix='.txt')

    # Read context 
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(programs_dir):
            filepath = os.path.join(programs_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as readfile:
                    outfile.write(readfile.read() + '\n')

    # Vector database
    loader = TextLoader(file_path=output_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    persist_path = os.path.join(tempfile.gettempdir(), f"{support_option.replace(' ', '_').lower()}_vector_store.parquet")
    vector_store = SKLearnVectorStore.from_documents(documents=docs, embedding=embeddings, persist_path=persist_path, serializer="parquet")
    
    # Input question 
    question = st.text_input("How can we assist you today?", "")
    
    # Generate response
    if st.button("Ask") and question:
        context = retrieve_document(question, vector_store)
        if context != "No relevant document found.":
            prompt = f'''
INSTRUCTION: You are an assistant to disadvantage people in delaware. Assist the user based on the following query, and use the context below.

QUERY: {question}

CONTEXT: {context}
'''
            answer = generate_text(prompt)
            st.text_area("Response", answer, height=300)
            st.session_state['context'] = context  #
            st.session_state['previous_answer'] = answer
            st.session_state['show_follow_up'] = True

    # Follow-up questions
    if st.session_state.get('show_follow_up', False):
        follow_up_question = st.text_input("Do you have any follow-up questions?", "")
        if st.button("Submit Follow-Up"):
            follow_up_context = st.session_state['context']  
            follow_up_prompt = f'''
INSTRUCTION: Answer the following follow-up query based on the initial context and response provided: {follow_up_question}

INITIAL CONTEXT: {follow_up_context}
INITIAL RESPONSE: {st.session_state['previous_answer']}
'''
            follow_up_answer = generate_text(follow_up_prompt)
            st.text_area("Follow-Up Response", follow_up_answer, height=300)



def forum_page():
    """Community Forum Page."""
    st.title("Community Forum")
    if 'forum_posts' not in st.session_state:
        st.session_state.forum_posts = [
            {"id": 1, "author": "John Smith", "content": "Just wanted to share my experience with the childcare support application process. It was daunting at first, but thanks to the guidance I found here and the clear instructions provided by the service, I was able to complete my application successfully.", "replies": []},
            {"id": 2, "author": "Jane Doe", "content": "Has anyone here applied for housing support recently? I'm looking for some tips on how to ensure my application is processed smoothly.", "replies": [{"author": "Responder Name", "content": "I applied last month and found that having all my documents prepared in advance was really helpful.", "replies": []}]},
            {"id": 3, "author": "Craig Johnson", "content": "Can anyone recommend education support programs for single parents? I'm trying to find something that can help me balance my studies while taking care of my kids.", "replies": []}
        ]

    new_post_content = st.text_input("Share your experiences, ask questions, and discuss with others!", key="new_post")
    if st.button("Post", key="btn_new_post"):
        new_post_id = len(st.session_state.forum_posts) + 1
        new_post = {"id": new_post_id, "author": "Your Name", "content": new_post_content, "replies": []}
        st.session_state.forum_posts.append(new_post)

    for post in st.session_state.forum_posts:
        with st.expander(f"{post['author']}: {post['content']}"):
            for reply in post['replies']:
                st.write(f"- {reply['author']}: {reply['content']}")
            reply_text = st.text_input("Reply to this post", key=f"reply_{post['id']}")
            if st.button("Reply", key=f"btn_reply_{post['id']}"):
                post['replies'].append({"author": "Your Name", "content": reply_text, "replies": []})
                st.experimental_rerun()


def setup_navigation():
    """Setup navigation for the app."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ['Support Services', 'Community Forum'])

    if page == 'Support Services':
        main_page()
    elif page == 'Community Forum':
        forum_page()

if __name__ == '__main__':
    st.session_state.clear()
    st.set_page_config(page_title="gov.AI.de", page_icon=":house_with_garden:")
    setup_navigation()