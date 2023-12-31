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
from google.generativeai.types import HarmCategory
from google.ai.generativelanguage import SafetySetting
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper, SerpAPIWrapper

DST_DIR = Path("output")
DST_DIR.mkdir(exist_ok=True, parents=True)

load_dotenv(find_dotenv())

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
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


class PipelineSingleton:
    """
    the local 7B models are very slow, so make sure to initialize them only once
    """

    _instance = None
    _pipeline = None

    def __new__(cls, model_name):
        if cls._instance is None:
            cls._instance = super(PipelineSingleton, cls).__new__(cls)
            cls._initialize_pipeline(model_name)

        return cls._instance

    @classmethod
    def _initialize_pipeline(cls, model_name):
        if cls._pipeline is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    @classmethod
    def get_pipeline(cls):
        return cls._pipeline


def transform_raw_answer(text_block):
    """
    Reformats the question-answer text block to reinforce the list aspect

    Args:
        - text_block (str): The text block to be reformatted.

    Returns:
        - str: The reformatted text block.
    """
    try:
        lines = text_block.split("\n")
        reformatted_lines = []

        for line in lines:
            if "Question:" in line or "Answer:" in line:
                reformatted_line = line.split(":", 1)[1].strip()
                reformatted_lines.append(reformatted_line)

        reformatted_text = "\n".join(reformatted_lines)
    except AttributeError as e:
        reformatted_text = ""
        logging.error(f"{e}")

    return reformatted_text


def research(chat_summary, agent, verbose=False):
    """
    response_content = responses[0].message.content
    """
    # palm.configure(api_key=palm_api_key)
    # researcher = palm.generate_text
    # researcher = model

    format_prompt = """
        Take the CHAT SUMMARY that follows and create a new bullet point list.
        Make sure the list contains the same information as the summary with a key difference --
        The new list items will take the form of questions that can be asked.
        The new list item will only contain questions about the facts and evidence from the original summary item.
        The new list item will not ask questions about the name attributed to the statement or argument.
        The new list item will only ask questions about facts and evidence.
        Opinions must be left out since they can not be proven.
        For example:
            - Given:
                * John
                   * argument: Advocated for American independence, asserting that the British monarchy has no rightful authority over the American colonies.
            - EXAMPLE BAD QUESTION
                   * Who was John?
            - EXAMPLE BAD QUESTION
                   * Did John advocate for indepdence?
            - EXAMPLE GOOD QUESTION
                   * Did the British monarchy have authority over the american colonies?

        The goal of this exercise is to search and find supporting detail to verify or disprove the facts.
        The questions must be short enough and in a format that is friendly for Wikipedia.
        The questions must contain enough information to produce a sensible Wikipedia search query.

        The new list will strictly take this form:
            * <list item question>?

        Do not add text before or after that format.

        CHAT SUMMARY:
    """
    format_prompt += chat_summary
    # sequences = researcher(
    #    prompt=format_prompt,
    #    safety_settings=[
    #        {
    #            "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
    #            "threshold": SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    #        },
    #    ],
    # )
    responses = respond(agent.client, format_prompt, model=agent.model_name)
    sequences = responses[0].message.content

    # summary_fmt = sequences.result.split("\n")
    summary_fmt = sequences.split("\n")

    if verbose:
        logging.info(f"summary fmt: {summary_fmt}")

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    # search = SerpAPIWrapper()

    # search_info = ""
    answers = []
    for summary in summary_fmt:
        try:
            info = wikipedia.run(summary, load_max_docs=1)
        except Exception as e:
            info = ""
            logging.error(f"{e}")

        # info = search.run(summary)
        # search_info += info

        # these can be really long
        # logging.info(f"wiki info: {info}")

        check_prompt = """
            Use the SUMMARY QUESTION that follows and the SEARCHED FACTS that follow.
            Use the SEARCHED FACTS to answer the SUMMARY QUESTION.
            Attach a brief one line explanation based on the SEARCHED FACTS.
            If the question is asking an opinion, respond that you only research facts and evidence.
            If the question is speculative, respond that you only research facts and evidence.
            If the question is subjective, respond that you only research facts and evidence.

            Your response will need to strictly adhere to this format:
                * Question: <SUMMARY QUESTION>?
                * Answer: <brief explanation>

        """
        check_prompt += "\nSUMMARY QUESTION\n"
        check_prompt += summary
        check_prompt += "\nSEARCHED FACTS:\n"
        check_prompt += info

        # sequences = researcher(
        #    prompt=check_prompt,
        #    safety_settings=[
        #        {
        #            "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
        #            "threshold": SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        #        },
        #    ],
        # )
        responses = respond(agent.client, check_prompt, model=agent.model_name)
        sequences = responses[0].message.content

        # answer = transform_raw_answer(sequences.result)
        answer = transform_raw_answer(sequences)
        answers.append(answer)

        if verbose:
            logging.info(f"summary question: {summary}")
            logging.info(f"sequences: {sequences}")
            logging.info(f"answer: {answer}")

    return answers


def summarize(history, chat_record, agent, verbose=False):
    """
    summarize the chat history to align with context window constraint
    """
    # palm.configure(api_key=palm_api_key)
    # summarizer = palm.generate_text

    # if the size of the chat history is too large, summarize it
    if len(chat_record) > 1:
        chat_history = " ".join(chat_record)

        postfix = """
        \n
        INSTRUCTIONS:
        Summarize the previous chat history into concise bullet points. It includes a moderator and two debaters.
        Significantly reduce the overall size.
        Capture main points from the arguments and examples.
        Prioritize preserving any facts that have been communicated.
        If a debater references something specific, like an historical event or fact, preserve this fact.
        The facts are important.
        Attribute the main points to the debater that introduced them.

        The chat format is:
            Name: <Name>
            Action: <Action> 
            Action Input: <Arguments, Examples>

        The summary format will look like:
            *moderator*:
                * statement:  
            *debater 1 name*
                * argument:
                * argument:  
            *debater 2 name*
                * argument:
                * argument:  
        """
        prompt = chat_history + postfix

        if verbose:
            logging.info(f"summarizer prompt: {prompt}")

        # sequences = summarizer(
        #    prompt=prompt,
        #    safety_settings=[
        #        {
        #            "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
        #            "threshold": SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        #        },
        #    ],
        # )
        # summary = sequences.result
        responses = respond(agent.client, prompt, model=agent.model_name)
        summary = responses[0].message.content

        if verbose:
            logging.info(f"Summarizing history: \n{summary}")

        return summary

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
        pipeline_singleton = PipelineSingleton(AgentModel.FALCON_7B_INSTRUCT.value)
        falcon_pipeline = pipeline_singleton.get_pipeline()

    if AgentModel.LLAMA_2_7B_CHAT_HF.value in models:
        tokenizer = AutoTokenizer.from_pretrained(AgentModel.LLAMA_2_7B_CHAT_HF.value)
        pipeline_singleton = PipelineSingleton(AgentModel.LLAMA_2_7B_CHAT_HF.value)
        llama_pipeline = pipeline_singleton.get_pipeline()

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
    """
    Description:
        - generate a prompt
        - insert values into the prompt template (taken from the config file)
        - values: chat history, role description (persona)
    """
    history = history if history else "This is the beginning of the debate."

    t = string.Template(template["prompts"]["prompt"])
    agent_di = get_agent(template, agent_name)
    role_desc = agent_di["role_description"]

    prompt = t.substitute(role_description=role_desc, chat_history=history)

    logging.info(f"prompt size: {len(prompt.split(' '))}")

    return prompt


def respond(client, prompt, model=None):
    # chatgpt
    if model in [AgentModel.CHATGPT_35_TURBO.value, AgentModel.CHATGPT_4.value]:
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(model=model, messages=messages)
        responses = completion.choices

        return responses

    # falcon
    elif model == AgentModel.FALCON_7B_INSTRUCT.value:
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

    # llama
    elif model == AgentModel.LLAMA_2_7B_CHAT_HF.value:
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

    # bison
    elif model == AgentModel.PALM_TEXT_BISON_001.value:
        sequences = client(
            prompt=prompt,
            safety_settings=[
                {
                    "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
                    "threshold": SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
            ],
        )

        message = Message(content=sequences.result)
        response = Response(message=message)

        return [response]
    else:
        raise ValueError("No model matched for response creation")


def write_output(
    history: str,
    summarized_history: str,
    debate: str,
    round_: int,
    dst_dir: Path = Path("output"),
    prefix: str = "",
):
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
    summarized_history = (
        summarized_history
        if isinstance(summarized_history, str)
        else "\n".join(summarized_history)
    )
    history_dst = dst_dir / f"{prefix}_{debate}_{round_}_{moderator}_history.txt"
    summarized_history_dst = (
        dst_dir / f"{prefix}_{debate}_{round_}_{moderator}_summarized_history.txt"
    )
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

        if model in [AgentModel.CHATGPT_35_TURBO.value, AgentModel.CHATGPT_4.value]:
            completion = client.chat.completions.create(model=model, messages=messages)
            resp = completion.choices[0].message.content
        elif model in [AgentModel.PALM_TEXT_BISON_001.value]:
            sequences = client(
                prompt=mono_prompt,
                safety_settings=[
                    {
                        "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
                        "threshold": SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                ],
            )

            message = Message(content=sequences.result)
            completion = Response(message=message)

            resp = completion.message.content
        elif model in [
            AgentModel.LLAMA_2_7B_CHAT_HF.value,
            AgentModel.FALCON_7B_INSTRUCT.value,
        ]:
            sequences = client(
                mono_prompt,
                max_length=10000,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
            )

            content = ""
            for seq in sequences:
                logging.info(f"{model} full response: \n{seq['generated_text']}")

                texts = seq["generated_text"].split("RESPONSE:")
                content = texts[-1]

            # form the response
            message = Message(content=content)
            response = Response(message=message)

            resp = response.message.content
        else:
            raise ValueError("Invalid model")

        # resp = completion.choices[0].message.content
        if verbose:
            # print(f"Initial Response:{response}\n")
            # print(f"Inner monologue: {resp}\n")
            logging.info(f"Initial Response: {response}")
            logging.info(f"Inner monologue: {resp}")
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
    # nltk.download("punkt")
    # nltk.download("gutenberg")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    palm_api_key = os.environ.get("PALM_API_KEY")

    ## Load debate templates and initialize history
    templates = load_templates(config_dir)
    debates = sorted(templates.keys())

    # Select debate; set toggles
    st.markdown("## Debate Settings")
    debate = st.selectbox("Select debate:", tuple(debates))
    n_rounds = st.select_slider(label="Number of debate rounds", options=range(5, 31))
    inner_monologue = st.checkbox("Use inner monologue?")
    inner_rounds = st.select_slider(
        label="Number of inner monologue rounds", options=range(1, 6)
    )
    template = templates[debate]
    summarization = st.checkbox("Reduce prompt size with chat history summarization")
    live_research = st.checkbox("Research statements as they are made")

    st.markdown("## Debate Logging")
    verbose = st.checkbox("Verbose inner monologue (print in local terminal)")
    verbose_summarization = st.checkbox("Verbose summarization")
    verbose_research = st.checkbox("Verbose research")

    logging.info(
        f"n_rounds: {n_rounds}, inner_monologue: {inner_monologue}, verbose: {verbose}, summarization: {summarization}"
    )

    # model API keys check
    if openai_api_key is None:
        logging.warning("OpenAI API Key is missing")
    if palm_api_key is None:
        logging.warning("PaLM API Key is missing")

    if palm_api_key is None and (summarization or live_research):
        raise ValueError("PaLM API needs to be configured for selected debate options")

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
    # this should help distil the prompt context size
    chat_record = []
    annotated_chat_record = []

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
        for round_ in range(n_rounds):
            if round_ == 0:
                logging.info(f"First round: {round_}")

                with st.chat_message(moderator):
                    message_placeholder = st.empty()
                    full_response = ""
                    summarized_history = ""

                    if summarization:
                        summarized_history = summarize(
                            history,
                            annotated_chat_record,
                            agent_clients[moderator],
                            verbose=verbose_summarization,
                        )
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
                    annotated_chat_record.append(response_content)

                    logging.info(f"response_content: {response_content}")

                st.session_state.messages.append(
                    {"role": moderator, "content": chat_response_content}
                )
                file_prefix = f"{timestamp}_{debate}_{n_rounds}rounds_{inner_monologue}innermono_{inner_rounds}innerrounds_{round_}round"
                with open(DST_DIR / f"{file_prefix}_history.txt", "w") as f:
                    f.write(history)
                with open(DST_DIR / f"{file_prefix}_summarized_history.txt", "w") as f:
                    f.write(summarized_history)

            # Last round the moderator decides who won
            elif round_ == n_rounds - 1:
                logging.info(f"Last round: {round_}")

                with st.chat_message(moderator):
                    message_placeholder = st.empty()
                    summarized_history = ""

                    if summarization:
                        summarized_history = summarize(
                            history,
                            annotated_chat_record,
                            agent_clients[moderator],
                            verbose=verbose_summarization,
                        )

                    # moderator probably wants the entire history
                    # if summarization:
                    #    prompt = make_prompt(template, moderator, summarized_history)
                    # else:
                    #    prompt = make_prompt(template, moderator, history)
                    prompt = make_prompt(template, moderator, history)

                    # experimental researcher agent
                    # if live_research:
                    #    research(summarized_history, agent_clients[moderator], verbose=verbose_research)

                    prompt += """
                    \nDecide who won the debate and explain why.
                    Provide a score of 0-100 for each debater and explain the reason for the score with an itemized break-down, score (0-20),
                    and explanation using the following criteria:
                        Organization and Clarity,
                        Use of Arguments,
                        Use of examples and facts,
                        Use of rebuttal,
                        Presentation Style.
                    Then give an overall score for each debater.
                    """

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
                    annotated_chat_record.append(response_content)

                    logging.info(f"response_content: {response_content}")

                st.session_state.messages.append(
                    {"role": moderator, "content": chat_response_content}
                )
                file_prefix = f"{timestamp}_last_{debate}_{n_rounds}rounds_{inner_monologue}innermono_{inner_rounds}innerrounds_{round_}round"
                with open(DST_DIR / f"{file_prefix}_history.txt", "w") as f:
                    f.write(history)
                with open(DST_DIR / f"{file_prefix}_summarized_history.txt", "w") as f:
                    f.write(summarized_history)
                with open(DST_DIR / f"{file_prefix}_chat_record.txt", "w") as f:
                    f.write("\n".join(chat_record))
            # Debate occurs between two participants otherwise
            else:
                logging.info(f"Round: {round_}")

                for debater in debaters:
                    with st.chat_message(debater):
                        if inner_monologue:
                            inner_placeholder = st.empty()

                        message_placeholder = st.empty()
                        summarized_history = ""

                        if summarization:
                            summarized_history = summarize(
                                history,
                                annotated_chat_record,
                                agent_clients[debater],
                                verbose=verbose_summarization,
                            )

                        if summarization:
                            prompt = make_prompt(template, debater, summarized_history)
                        else:
                            prompt = make_prompt(template, debater, history)

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
                            response_content = responses[inner_rounds]
                            inner_placeholder.markdown(
                                f"Inner monologue: {inner}\n\n---\n\n"
                            )
                            if verbose:
                                logging.info(f"verbose: {verbose} --------")
                        else:
                            responses = respond(
                                agent_clients[debater].client,
                                prompt,
                                model=agent_clients[debater].model_name,
                            )
                            response_content = responses[0].message.content

                        if live_research:
                            partial_summary = summarize(
                                "",
                                ["", response_content],
                                agent_clients[debater],
                                verbose=verbose_summarization,
                            )
                            items = research(
                                partial_summary,
                                agent_clients[debater],
                                verbose=verbose_research,
                            )

                            research_placeholder = st.empty()
                            research_placeholder.markdown(
                                f"<p style='color: orange;'>Research:</p>",
                                unsafe_allow_html=True,
                            )
                            for item in items:
                                st.markdown(
                                    f"<p style='color: orange;'> - {item}</p>",
                                    unsafe_allow_html=True,
                                )

                        # response_content = responses[0].message.content
                        chat_response_content = response_content.split("Action Input:")[
                            -1
                        ]
                        message_placeholder.markdown(chat_response_content)
                        history = append_to_history(history, response_content)

                        chat_record.append(chat_response_content)
                        annotated_chat_record.append(response_content)

                        logging.info(f"chat response content: {chat_response_content}")

                    st.session_state.messages.append(
                        {"role": debater, "content": chat_response_content}
                    )
                file_prefix = f"{timestamp}_{debate}_{n_rounds}rounds_{inner_monologue}innermono_{inner_rounds}innerrounds_{round_}round"
                with open(DST_DIR / f"{file_prefix}_history.txt", "w") as f:
                    f.write(history)
                with open(DST_DIR / f"{file_prefix}_summarized_history.txt", "w") as f:
                    f.write(summarized_history)
