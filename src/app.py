#!/usr/bin/env python3

# Standard library imports
import asyncio
from dataclasses import dataclass
import datetime
from enum import Enum
import logging
import os
import re

from pathlib import Path
import string
import time
from typing import Any
import yaml

# Third party imports
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import transformers
import torch
import nltk
import google.generativeai as palm

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DST_DIR = Path("output")
DST_DIR.mkdir(exist_ok=True, parents=True)

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
    CHATGPT_4 = "gpt-4"
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
        # openai API gpt 3.5 turbo
        elif model == AgentModel.CHATGPT_35_TURBO.value:
            client = OpenAI(api_key=openai_api_key)
            agent_clients[agent_name] = Agent(
                name=agent_name,
                model_name=model,
                client=client,
            )
        # openai API gpt 4 API
        elif model == AgentModel.CHATGPT_4.value:
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


def write_output(history: str, summarized_history: str, debate: str, round_: int, dst_dir: Path=Path("output"), prefix: str=""):
    """
    Write history and summarized history after each agent speaks.
    Arguments:
        history: Chat history.
        summarized_history: Summarized chat history.
        debate: Name of debate.
        round_: Round of debate.
        dst_dir: Output directory to save text files.
        timestamp: Optional prefixed timestamp.
    """

    dst_dir = dst_dir if isinstance(dst_dir, Path) else Path(dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)
    history = history if isinstance(history, str) else "\n".join(history)
    summarized_history = summarized_history if isinstance(summarized_history, str) else "\n".join(summarized_history)
    history_dst = dst_dir / f"{prefix}_{debate}_{round_}_{moderator}_history.txt"
    summarized_history_dst = dst_dir / f"{prefix}_{debate}_{round_}_{moderator}_summarized_history.txt"
    with open(history_dst, "w") as f:
        f.write("".join(history))
    with open(summarized_history_dst, "w") as f:
        f.write("".join(summarized_history))


def make_responses(
    client, template, agent_name, prompt, model=None, rounds=1, verbose=True
):
    if model is None:
        model = "gpt-3.5-turbo"
    responses = []
    response = respond(client, prompt, model)[
        0
    ].message.content  # initial response content
    responses.append(response)
    resp_ret = ""
    for i in range(rounds):  # use inner monologue to improve response
        agent_di = get_agent(template, agent_name)
        inner_prompt = ""
        try:
            inner_prompt = agent_di["inner_prompt"]
        except:
            inner_prompt = "Given the above prompt and response explain how you can improve the response, and then provide a revised response in the same format as the original response.  Ensure that the revised response does not repeat any points that have already been made and that the name and identity in the initial response does not change.  Ensure that the argument remains consistent and under 100 words."

        mono_prompt = f"{prompt}\nResponse:{response}\n{agent_di['inner_prompt']}"
        messages = [{"role": "user", "content": mono_prompt}]

        logging.info(f">>> mono prompt: {inner_prompt} <<<")

        if model in [AgentModel.CHATGPT_35_TURBO.value, AgentModel.CHATGPT_4.value]:
            completion = client.chat.completions.create(model=model, messages=messages)
            resp = completion.choices[0].message.content
            logging.info(f">>> inner prompt resp: {resp}; \nmodel: {model} <<<")
        elif model in [AgentModel.PALM_TEXT_BISON_001.value]:
            sequences = client(prompt=mono_prompt)

            message = Message(content=sequences.result)
            completion = Response(message=message)

            resp = completion.message.content
            logging.info(f">>> inner prompt resp: {resp}; \nmodel: {model} <<<")
        else:
            raise ValueError("Invalid model")

        #resp = completion.choices[0].message.content
        if verbose:
            print(f"Initial Response:{response}\n")
            print(f"Inner monologue: {resp}\n")
        m = [resp]  # re.findall('Action Input: .*$', resp)
        responses.append(m[0])
        response = m[0]
        resp = re.findall("(^.*)", resp)[0]
        resp_ret = resp_ret + f"Round {i+1}: {resp}\n\n"
    return resp_ret, responses


def choose_response(responses, client, prompt, model=None, rounds=1):
    """
    This function is deprecated and only here for reference purposes
    """
    """if model is None:
        model = "gpt-3.5-turbo"
    choose_prompt = f"{prompt} Given the following responses, return the number, and only the number of which response below is the best.\n"
    for i in range(rounds+1):
        choose_prompt+=f"{i}. {responses[i]}\n"
    print(f"\n\nPrompt: {choose_prompt}")
    messages = [{"role":"user", "content": prompt}]
    completion = client.chat.completions.create(model=model, messages=messages)
    print(completion.choices[0].message.content)"""
    return responses[rounds]


if __name__ == "__main__":
    config_dir = Path("./configs")
    logging.info(f"Config path {config_dir}")

    # set up nltk for chat history summarization
    nltk.download("punkt")
    nltk.download("gutenberg")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    palm_api_key = os.environ.get("PALM_API_KEY")

    ## Load debate templates and initialize history
    templates = load_templates(config_dir)
    debates = sorted(templates.keys())

    # Select debate
    debate = st.selectbox("Select debate:", tuple(debates))
    n_rounds = st.select_slider(label="Number of debate rounds", options=range(5, 31))
    inner_monologue = st.checkbox("Use inner monologue?")
    verbose = st.checkbox("Verbose (print in local terminal)")
    inner_rounds = st.select_slider(
        label="Number of inner monologue rounds", options=range(1, 6)
    )
    template = templates[debate]
    summarization = st.checkbox("Reduce prompt size with chat history summarization")

    logging.info(f"n_rounds: {n_rounds}, inner_monologue: {inner_monologue}, verbose: {verbose}, summarization: {summarization}")

    if openai_api_key is None:
        logging.warning("OpenAI API Key is missing")
    if palm_api_key is None:
        logging.warning("PaLM API Key is missing")

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
    if "timestamp" not in locals():
        timestamp = str(datetime.datetime.now())[:-10].replace(":", "-")

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

                    if summarization:
                        prompt = make_prompt(template, moderator, summarized_history)
                    else:
                        prompt = make_prompt(template, moderator, history)

                    responses = respond(
                        agent_clients[moderator].client,
                        prompt,
                        model=agent_clients[moderator].model_name,
                    )

                    response_content = responses[0].message.content
                    chat_response_content = response_content.split("Action Input:")[-1]

                    if moderator == "EMH":
                        message_placeholder.markdown(
                            "Please state the nature of the moderation emergency.\n\n"
                            + chat_response_content
                        )  # joke for Sam's benefit
                    else:
                        message_placeholder.markdown(chat_response_content)

                    message_placeholder.markdown(chat_response_content)
                    history = append_to_history(history, response_content)
                    chat_record.append(chat_response_content)
                    logging.info(f"response_content: {response_content}")
                st.session_state.messages.append(
                    {"role": moderator, "content": chat_response_content}
                )
                file_prefix = f"{timestamp}_{debate}_{n_rounds}rounds_{inner_monologue}innermono_{inner_rounds}innerrounds_{round_}round"
                with open(DST_DIR / f"{file_prefix}_history.txt", "w") as f: f.write(history)
                with open(DST_DIR / f"{file_prefix}_summarized_history.txt", "w") as f: f.write(summarized_history)


            # Last round the moderator decides who won
            elif round_ == n_rounds - 1:
                logging.info(f"Last round: {round_}")

                with st.chat_message(moderator):
                    message_placeholder = st.empty()
                    summarized_history = summarize(history, chat_record)

                    if summarization:
                        prompt = make_prompt(template, moderator, summarized_history)
                    else:
                        prompt = make_prompt(template, moderator, history)

                    prompt += "\nDecide who won the debate and explain why.  Provide a score of 0-100 for each debater and explain the reason for the score with an itemized break-down, score (0-20), and explanation using the following criteria: Organization and Clarity, Use of Arguments, Use of examples and facts, Use of rebuttal, Presentation Style.  Then give an overall score for each debater."

                    responses = respond(
                        agent_clients[moderator].client,
                        prompt,
                        model=agent_clients[moderator].model_name,
                    )

                    response_content = responses[0].message.content
                    chat_response_content = response_content.split("Action Input:")[-1]
                    message_placeholder.markdown(chat_response_content)
                    history = append_to_history(history, response_content)

                    chat_record.append(chat_response_content)
                    logging.info(f"response_content: {response_content}")

                st.session_state.messages.append(
                    {"role": moderator, "content": chat_response_content}
                )
                file_prefix = f"{timestamp}_last_{debate}_{n_rounds}rounds_{inner_monologue}innermono_{inner_rounds}innerrounds_{round_}round"
                with open(DST_DIR / f"{file_prefix}_history.txt", "w") as f: f.write(history)
                with open(DST_DIR / f"{file_prefix}_summarized_history.txt", "w") as f: f.write(summarized_history)
                with open(DST_DIR / f"{file_prefix}_chat_record.txt", "w") as f: f.write("\n".join(chat_record))
            # Debate occurs between two participants otherwise
            else:
                logging.info(f"Round: {round_}")

                for debater in debaters:
                    with st.chat_message(debater):
                        if inner_monologue:
                            inner_placeholder = st.empty()

                        message_placeholder = st.empty()
                        summarized_history = summarize(history, chat_record)

                        if summarization:
                            prompt = make_prompt(template, moderator, summarized_history)
                        else:
                            prompt = make_prompt(template, moderator, history)

                        if inner_monologue:
                            inner, responses = make_responses(
                                agent_clients[debater].client,
                                template,
                                debater,
                                prompt,
                                rounds=inner_rounds,
                                verbose=verbose,
                                model=agent_clients[debater].model_name,
                            )  # Create three responses based off of an inner monologue
                            response_content = responses[
                                inner_rounds
                            ]  # choose_response(responses, agent_clients[debater], prompt, rounds=inner_rounds)  # Choose the strongest of the three responses
                            inner_placeholder.markdown(
                                f"Inner monologue: {inner}\n\n---\n\n"
                            )
                            if verbose:
                                print("---")
                        else:
                            # responses = respond(agent_clients[debater], prompt)
                            responses = respond(
                                agent_clients[debater].client,
                                prompt,
                                model=agent_clients[debater].model_name,
                            )
                            response_content = responses[0].message.content

                        # responses = respond(
                        #    agent_clients[debater].client,
                        #    prompt,
                        #    model=agent_clients[debater].model_name,
                        # )

                        # response_content = responses[0].message.content
                        chat_response_content = response_content.split("Action Input:")[
                            -1
                        ]
                        message_placeholder.markdown(chat_response_content)
                        history = append_to_history(history, response_content)

                        chat_record.append(chat_response_content)
                        logging.info(f"chat response content: {chat_response_content}")

                    st.session_state.messages.append(
                        {"role": debater, "content": chat_response_content}
                    )
                file_prefix = f"{timestamp}_{debate}_{n_rounds}rounds_{inner_monologue}innermono_{inner_rounds}innerrounds_{round_}round"
                with open(DST_DIR / f"{file_prefix}_history.txt", "w") as f: f.write(history)
                with open(DST_DIR / f"{file_prefix}_summarized_history.txt", "w") as f: f.write(summarized_history)
