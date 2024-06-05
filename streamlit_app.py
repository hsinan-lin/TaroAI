import os
import streamlit as st
from LLM import TaiwaneseDictionaryRetreiver
# Theming at: ~/.streamlit/config.toml
# Build a basic LLM chat app: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps


def header():
    st.markdown(
        """
        <h2 style='
        text-align: center;
        color: #6E75A8; 
        background-color: white;
        box-shadow: 0 8px 0 #F2F2F7;
        border-radius: 20px;
        border: 1px solid #E6E6EB;
        margin: 0 auto;'>
            芋圓台語AI字典
        </h2>
        <p></p>
        <h5 style='
        text-align: center; 
        color: #8D91C7;'>
            🤖 AI幫你速查台語 🤖
        </h5>
        """, 
        unsafe_allow_html=True
    )
    cols = st.columns([0.15, 0.15, 0.3, 0.15, 0.15])
    with cols[1]:
        st.image('./SittingOnAppStoreLogo.svg')
    with cols[2]:
        st.image('./pile.png')
    with cols[3]:
        st.image('./SittingOnAppStoreLogo.svg')

def chatarea():    
    # Clear results
    cols = st.columns([0.15, 0.2, 0.4, 0.15])
    with cols[-3]:
        search_mode_selected = st.selectbox(
            "Select search mode",
            (
                "一般查詢",
                "音檔查詢"
            ),
            index=0,
            placeholder="Select search mode",
            label_visibility="collapsed"
        )
    with cols[-2]:
        model_selected = st.selectbox(
            "Select a model",
            [
                # HuggingFace
                "mistralai/Mistral-7B-Instruct-v0.2",
                "ChatGPT",
                # Ollama / Local
                "llama3",
                # "cwchang/llama3-taide-lx-8b-chat-alpha1",
                "wangshenzhi/llama3-8b-chinese-chat-ollama-q4",
                # "wangrongsheng/taiwanllm-7b-v2.1-chat",
                # "mistral",
            ],
            index=1,
            placeholder="Select a model",
            label_visibility="collapsed"
        )
    with cols[-1]:
        if st.button("清除", type="primary", use_container_width=True):
            st.session_state.messages = []
    
    # Initialize LLM
    llm = TaiwaneseDictionaryRetreiver(model=model_selected)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            response = message["content"]
            if message["role"] == "assistant":
                if search_mode_selected == "一般查詢":
                    st.markdown(response)
                else:
                    answer = response.get("kanji", "無")
                    lomajis = response.get("lomajis", [])
                    wordIDs = response.get("wordIDs", [])
                    st.markdown(f"""
                        回答：
                        {answer}
                        
                        羅馬字：
                        {" / ".join(lomajis)}
                        
                        音檔：
                    """)
                    for id in wordIDs:
                        path = f"./AllKipAudio/{id}.mp3"
                        if os.path.isfile(path):
                            st.audio(path)
            else:
                st.markdown(response)

    # React to user input
    if prompt := st.chat_input("要查什麼台語詞？"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            
            # Get LLM's response
            print("start")
            if search_mode_selected == "一般查詢":
                response = st.write_stream(llm.query(prompt + "的台語"))
                print(response)
            else:
                response = llm.query_json(prompt + "的台語")
                print(response)
                answer = response.get("kanji", "無")
                lomajis = response.get("lomajis", [])
                wordIDs = response.get("wordIDs", [])
                st.markdown(f"""
                    回答：
                    {answer}
                    
                    羅馬字：
                    {" / ".join(lomajis)}
                    
                    音檔：
                """)
                for id in wordIDs:
                    path = f"./AllKipAudio/{id}.mp3"
                    if os.path.isfile(path):
                        st.audio(path)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
if __name__ == "__main__":
    header()
    chatarea()

