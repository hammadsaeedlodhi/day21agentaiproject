import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch
import wikipedia
import arxiv
import google.generativeai as genai
from openai import OpenAI

# --------------------------
# 1. Load environment
# --------------------------
load_dotenv()

st.set_page_config(page_title="🧠 Multi-Model ReAct Agent", page_icon="🤖")
st.title("🤖 Multi-Model ReAct Agent (Groq + OpenAI + Hugging Face + Gemini)")

# --------------------------
# Sidebar: Model provider, key, and model selection
# --------------------------
st.sidebar.header("⚙️ Settings")

# Select model provider
provider = st.sidebar.selectbox(
    "Select Model Provider",
    ["--Select--", "Groq", "OpenAI", "Hugging Face", "Gemini"],
    index=0
)

# Initialize variables
groq_key = openai_key = hf_key = gemini_key = model_name = ""

# Show API key + model based on provider
if provider == "Groq":
    groq_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password") or os.getenv("GROQ_API_KEY", "")
    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        ["llama-3.1-8b-instant", "mixtral-8x7b"],
        index=0
    )

elif provider == "OpenAI":
    openai_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password") or os.getenv("OPENAI_API_KEY", "")
    model_name = st.sidebar.selectbox(
        "Select OpenAI Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )

elif provider == "Hugging Face":
    hf_key = st.sidebar.text_input("🔑 Enter Hugging Face API Key", type="password") or os.getenv("HUGGINGFACE_API_KEY", "")
    model_name = st.sidebar.selectbox(
        "Select Hugging Face Model",
        ["google/flan-t5-base", "tiiuae/falcon-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.2"],
        index=0
    )

elif provider == "Gemini":
    gemini_key = st.sidebar.text_input("🔑 Enter Gemini API Key", type="password") or os.getenv("GEMINI_API_KEY", "")
    model_name = st.sidebar.selectbox(
        "Select Gemini Model",
        ["gemini-pro", "models/gemini-1.5-flash", "models/gemini-1.5-pro"],
        index=0
    )

# Reasoning steps
max_steps = st.sidebar.slider("Max reasoning steps", 1, 6, 3)

st.markdown("""
This AI Agent follows the **ReAct pattern** (*Think → Act → Observe → Conclude*)  
and can search across: 🌐 Web, 📘 Wikipedia, 📄 Arxiv, 🎬 YouTube
""")

# --------------------------
# 2. Tool Functions
# --------------------------
def tool_web_search(query, k=4):
    with DDGS() as ddg:
        results = ddg.text(query, region="us-en", max_results=k)
        lines = [f"- {r.get('title','')} ({r.get('href','')})\n{r.get('body','')}" for r in results]
        return "\n".join(lines) if lines else "No web results found."

def tool_wikipedia(query, sentences=2):
    try:
        wikipedia.set_lang("en")
        pages = wikipedia.search(query, results=1)
        if not pages:
            return "No Wikipedia page found."
        summary = wikipedia.summary(pages[0], sentences=sentences)
        return f"Wikipedia: {pages[0]}\n{summary}"
    except Exception as e:
        return f"Wikipedia error: {e}"

def tool_arxiv(query):
    try:
        search = arxiv.Search(query=query, max_results=1, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
        if not results:
            return "No Arxiv paper found."
        paper = results[0]
        snippet = (paper.summary or "").replace("\n"," ")[:400]
        return f"arxiv: {paper.title}\nLink: {paper.entry_id}\n{snippet}..."
    except Exception as e:
        return f"Arxiv error: {e}"

def tool_youtube(query):
    try:
        videos = VideosSearch(query, limit=3).result()["result"]
        if not videos:
            return "No YouTube videos found."
        lines = [f"🎬 {v['title']} ({v['link']}) - {v['channel']['name']}" for v in videos]
        return "\n".join(lines)
    except Exception as e:
        return f"YouTube error: {e}"

# --------------------------
# 3. ReAct Prompt
# --------------------------
SYSTEM_PROMPT = """
You are a helpful AI research assistant with access to 4 tools:
1️⃣ WebSearch
2️⃣ Wikipedia
3️⃣ Arxiv
4️⃣ YouTube

Follow this reasoning pattern strictly:

Thought: what you are thinking next
Action: which tool you will use (WebSearch, Wikipedia, Arxiv, YouTube)
Action Input: your search query
Observation: (tool output will be provided here)

Repeat until you can confidently give a conclusion.
Then write:
Final Answer: <short, clear response>
"""

ACTION_RE = re.compile(r"^Action:\s*(WebSearch|Wikipedia|Arxiv|YouTube)", re.I)
INPUT_RE = re.compile(r"^Action Input:\s*(.*)", re.I)

# --------------------------
# 4. LLM client selector
# --------------------------
def generate_response(prompt):
    """Route prompt to the right LLM provider."""
    try:
        if provider == "Groq":
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            return resp.choices[0].message.content

        elif provider == "OpenAI":
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            return resp.choices[0].message.content

        elif provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {hf_key}"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400}}
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_name}",
                headers=headers, json=payload
            )
            data = response.json()
            return data[0]["generated_text"] if isinstance(data, list) else str(data)

        elif provider == "Gemini":
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text

        else:
            return "❌ No valid model provider selected."
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# --------------------------
# 5. Mini ReAct Agent
# --------------------------
def mini_agent(question, max_iters=3):
    transcript = [f"User Question: {question}"]
    observation = None

    for step in range(1, max_iters + 1):
        convo = SYSTEM_PROMPT + "\n" + "\n".join(transcript)
        if observation:
            convo += f"\nObservation: {observation}"

        text = generate_response(convo)

        with st.expander(f"🧩 Step {step}", expanded=False):
            st.write(text)

        if "Final Answer:" in text:
            return text.split("Final Answer:", 1)[1].strip()

        action, action_input = None, None
        for line in text.splitlines():
            if ACTION_RE.match(line):
                action = ACTION_RE.match(line).group(1).title()
            if INPUT_RE.match(line):
                action_input = INPUT_RE.match(line).group(1).strip()

        if not action or not action_input:
            return "Could not understand next step."

        # Choose tool
        if action == "Websearch":
            observation = tool_web_search(action_input)
        elif action == "Wikipedia":
            observation = tool_wikipedia(action_input)
        elif action == "Arxiv":
            observation = tool_arxiv(action_input)
        elif action == "Youtube":
            observation = tool_youtube(action_input)
        else:
            observation = f"Unknown tool: {action}"

        transcript.append(f"Thought: I will use {action}.")
        transcript.append(f"Action: {action}")
        transcript.append(f"Action Input: {action_input}")
        transcript.append(f"Observation: {observation}")

    summary = generate_response("\n".join(transcript) + "\nSummarize briefly in English.")
    return summary

# --------------------------
# 6. Streamlit Chat UI
# --------------------------
query = st.chat_input("💬 Ask me anything...")

if query:
    st.chat_message("user").write(query)

    # API key validation
    if provider == "Groq" and not groq_key:
        st.error("Please enter your Groq API Key.")
    elif provider == "OpenAI" and not openai_key:
        st.error("Please enter your OpenAI API Key.")
    elif provider == "Hugging Face" and not hf_key:
        st.error("Please enter your Hugging Face API Key.")
    elif provider == "Gemini" and not gemini_key:
        st.error("Please enter your Gemini API Key.")
    else:
        with st.spinner("🤔 Thinking..."):
            answer = mini_agent(question=query, max_iters=max_steps)
        st.chat_message("assistant").write(answer)
