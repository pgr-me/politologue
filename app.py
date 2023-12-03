#!/usr/bin/env python3

# Standard library imports
import asyncio
import string
import time
import yaml
import os
import logging

from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Any

# Third party imports
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import transformers
import torch
import nltk
import google.generativeai as palm

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv(find_dotenv())

# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class AgentModel(Enum):
    """
    available API's and local quantized models
    """

    CHATGPT_35_TURBO = "gpt-3.5-turbo"
    FALCON_7B_INSTRUCT = "tiiuae/falcon-7b-instruct"
    LLAMA_2_7B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
    PALM_TEXT_BISON_001 = "text-bison-001"


@dataclass
class Agent:
    """
    the agent has several attributes, so this class attempts to group them for easy access
    """

    name: str
    model_name: str
    client: Any


@dataclass
class Message:
    """
    the llm generates a sequence response that can be stored here in a format that is expected downstream
    similar to open AI's API response
    """

    content: str


@dataclass
class Response:
    """
    the agent response object that mimics OpenAI API's response format
    """

    message: Message


def chunk_text(text, chunk_size=500):
    """
    given a text separate the text into chunks for the language model context window constraint
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size or len(current_chunk) == 0:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "

    chunks.append(current_chunk)

    return chunks


def summarize(history, chat_record):
    """
    summarize the chat history to align with context window constraint
    """
    device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline(
        "summarization", model="pszemraj/led-large-book-summary", device=device
    )

    # if the size of the chat history is too large, summarize it
    # include the last portion of chat for continuity
    if len(chat_record) > 3:
        chunks = chunk_text(history, chunk_size=1024)
        summaries = [
            summarizer(chunk, max_length=100)[0]["summary_text"] for chunk in chunks
        ]
        summary = " ".join(summaries)
        summary = "\nDebate Summary:\n" + summary

        chat = " ".join(chat_record[-2:])
        chat = "\nMost Recent Chat History:\n" + chat
        summary = summary + chat

        logging.info(f"Summarizing history: \n{summary}")

    return history


def append_to_history(old_history, new_history):
    if not old_history:
        return new_history
    return f"{old_history}\n\n{new_history}"


def compress(client, text, model=None, prompt=None):
    if model is None:
        model = AgentModel.CHATGPT_35_TURBO.value
    if prompt is None:
        prompt = """Compress the following chat so much that maybe a human doesn't understand it but you do."""
    prompt = f"{prompt}\n{text}"
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(model=model, messages=messages)
    compressions = completion.choices
    return compressions


def get_agent(template, agent_name):
    agent_di = [x for x in template["agents"] if x["name"] == agent_name][0]
    return agent_di


def get_moderator(template):
    li = [x["name"] for x in template["agents"] if x["moderator"]]
    assert len(li) == 1, "There can be only one ... moderator."
    return li[0]


def instantiate_agents(template, openai_api_key=None, palm_api_key=None):
    logging.info(f"Instantiating agents")

    tokenizer = None
    falcon_pipeline = None
    llama_pipeline = None
    models = [agent["llm"]["model"] for agent in template["agents"]]

    # set up falcon if one of the agents requires it
    if AgentModel.FALCON_7B_INSTRUCT.value in models:
        tokenizer = AutoTokenizer.from_pretrained(AgentModel.FALCON_7B_INSTRUCT.value)
        falcon_pipeline = transformers.pipeline(
            "text-generation",
            model=AgentModel.FALCON_7B_INSTRUCT.value,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if AgentModel.LLAMA_2_7B_CHAT_HF.value in models:
        tokenizer = AutoTokenizer.from_pretrained(AgentModel.LLAMA_2_7B_CHAT_HF.value)
        llama_pipeline = transformers.pipeline(
            "text-generation",
            model=AgentModel.LLAMA_2_7B_CHAT_HF.value,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # set up each agent with a llm
    agent_clients = {}
    for agent_data in template["agents"]:
        agent_name = agent_data["name"]
        model = str(agent_data["llm"]["model"])

        # falcon
        if model == AgentModel.FALCON_7B_INSTRUCT.value:
            agent_clients[agent_name] = Agent(
                name=agent_name,
                model_name=model,
                client=falcon_pipeline,
            )
        # llama
        elif model == AgentModel.LLAMA_2_7B_CHAT_HF.value:
            agent_clients[agent_name] = Agent(
                name=agent_name,
                model_name=model,
                client=llama_pipeline,
            )
        # palm bison
        elif model == AgentModel.PALM_TEXT_BISON_001.value:
            palm.configure(api_key=palm_api_key)
            agent_clients[agent_name] = Agent(
                name=agent_name,
                model_name=model,
                client=palm.generate_text,
            )
        # chat gpt API
        elif model == AgentModel.CHATGPT_35_TURBO.value:
            client = OpenAI(api_key=openai_api_key)
            agent_clients[agent_name] = Agent(
                name=agent_name,
                model_name=model,
                client=client,
            )
        else:
            raise ValueError(f"No model configuration for {model}")

    return agent_clients


def load_templates(config_dir):
    templates = {}
    template_srcs = sorted(config_dir.glob("*.yaml"))
    for template_src in template_srcs:
        with open(template_src, "r") as f:
            template = yaml.safe_load(f)
            name = template["name"]
            templates[name] = template
    return templates


def make_prompt(template, agent_name, history):
    history = history if history else "This is the beginning of the debate."

    t = string.Template(template["prompts"]["prompt"])
    agent_di = get_agent(template, agent_name)
    role_desc = agent_di["role_description"]

    prompt = t.substitute(role_description=role_desc, chat_history=history)

    return prompt


def respond(client, prompt, model=None):
    if model is None:
        model = AgentModel.CHATGPT_35_TURBO.value

    if model == AgentModel.CHATGPT_35_TURBO.value:
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(model=model, messages=messages)
        responses = completion.choices

        return responses

    if model == AgentModel.FALCON_7B_INSTRUCT.value:
        sequences = client(
            prompt, max_length=10000, do_sample=True, top_k=10, num_return_sequences=1
        )

        content = ""
        for seq in sequences:
            logging.info(f"{model} full response: \n{seq['generated_text']}")

            texts = seq["generated_text"].split("RESPONSE:")
            content = texts[-1]

        # form the response
        message = Message(content=content)
        response = Response(message=message)

        return [response]

    if model == AgentModel.LLAMA_2_7B_CHAT_HF.value:
        sequences = client(
            prompt, max_length=10000, do_sample=True, top_k=10, num_return_sequences=1
        )

        content = ""
        for seq in sequences:
            logging.info(f"{model} full response: \n{seq['generated_text']}")

            texts = seq["generated_text"].split("RESPONSE:")
            content = texts[-1]

        # form the response
        message = Message(content=content)
        response = Response(message=message)

        return [response]

    if model == AgentModel.PALM_TEXT_BISON_001.value:
        sequences = client(prompt=prompt)

        message = Message(content=sequences.result)
        response = Response(message=message)

        return [response]

    raise ValueError("No response")


if __name__ == "__main__":
    config_dir = Path("./configs")
    logging.info(f"Config path {config_dir}")

    # set up nltk for chat history summarization
    nltk.download("punkt")
    nltk.download("gutenberg")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    palm_api_key = os.environ.get("PALM_API_KEY")

    if openai_api_key is None:
        logging.warning("OpenAI API Key is missing")

    if palm_api_key is None:
        logging.warning("PaLM API Key is missing")

    # Load debate templates and initialize history
    templates = load_templates(config_dir)
    debates = sorted(templates.keys())

    # Select debate
    debate = st.selectbox("Select debate:", tuple(debates))
    template = templates[debate]

    # Instantiate agent clients and history
    agent_clients = instantiate_agents(
        template,
        openai_api_key=openai_api_key,
        palm_api_key=palm_api_key,
    )
    agent_names = list(agent_clients.keys())

    logging.info(f"agent_clients: {agent_clients}")

    if "history" not in locals():
        history = ""

    # keep a chat record to help with summarization
    # the latest literal responses will be appended to a summary
    # this should help distil the prompt context size
    chat_record = []

    # if round_ not in locals():
    #    round_ = 0
    moderator = get_moderator(template)
    debaters = [x for x in agent_names if x != moderator]

    n_rounds = 5
    # Set default model
    logging.info(f"Session state: {st.session_state}")
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = AgentModel.CHATGPT_35_TURBO.value

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("Start Debate"):
        # st.session_state.messages.append({"role": "user", "content": prompt})
        for round_ in range(n_rounds):
            if round_ == 0:
                logging.info(f"First round: {round_}")

                with st.chat_message(moderator):
                    message_placeholder = st.empty()
                    full_response = ""
                    summarized_history = summarize(history, chat_record)

                    prompt = make_prompt(template, moderator, summarized_history)

                    responses = respond(
                        agent_clients[moderator].client,
                        prompt,
                        model=agent_clients[moderator].model_name,
                    )

                    response_content = responses[0].message.content
                    chat_response_content = response_content.split("Action Input:")[-1]
                    message_placeholder.markdown(chat_response_content)
                    history = append_to_history(history, response_content)

                    chat_record.append(response_content)

                st.session_state.messages.append(
                    {"role": moderator, "content": chat_response_content}
                )

            # Last round the moderator decides who won
            elif round_ == n_rounds - 1:
                logging.info(f"Last round: {round_}")

                with st.chat_message(moderator):
                    message_placeholder = st.empty()
                    summarized_history = summarize(history, chat_record)

                    prompt = make_prompt(template, moderator, summarized_history)
                    prompt += "\nDecide who won the debate and explain why."

                    responses = respond(
                        agent_clients[moderator].client,
                        prompt,
                        model=agent_clients[moderator].model_name,
                    )

                    response_content = responses[0].message.content
                    chat_response_content = response_content.split("Action Input:")[-1]
                    message_placeholder.markdown(chat_response_content)
                    history = append_to_history(history, response_content)

                    chat_record.append(response_content)

                st.session_state.messages.append(
                    {"role": moderator, "content": chat_response_content}
                )
            # Debate occurs between two participants otherwise
            else:
                logging.info(f"Round: {round_}")

                for debater in debaters:
                    with st.chat_message(debater):
                        message_placeholder = st.empty()
                        summarized_history = summarize(history, chat_record)

                        prompt = make_prompt(template, debater, summarized_history)

                        responses = respond(
                            agent_clients[debater].client,
                            prompt,
                            model=agent_clients[debater].model_name,
                        )

                        response_content = responses[0].message.content
                        chat_response_content = response_content.split("Action Input:")[
                            -1
                        ]
                        message_placeholder.markdown(chat_response_content)
                        history = append_to_history(history, response_content)

                        chat_record.append(response_content)

                    st.session_state.messages.append(
                        {"role": debater, "content": chat_response_content}
                    )
