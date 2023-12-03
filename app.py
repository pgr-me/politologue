#!/usr/bin/env python3

# Standard library imports
import asyncio
from pathlib import Path
import string
import time
import yaml
import re

# Third party imports
from openai import OpenAI
import streamlit as st
import os


def append_to_history(old_history, new_history):
    if not old_history:
        return new_history
    return f"{old_history}\n\n{new_history}"


def compress(client, text, model=None, prompt=None):
    if model is None:
        model = "gpt-3.5-turbo"
    if prompt is None:
        prompt = """Compress the following chat so much that maybe a human doesn't understand it but you do."""
    prompt = f"{prompt}\n{text}"
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(model=model, messages=messages)
    compressions = completion.choices
    return compressions


def get_agent(template, agent_name):
    agent_di = [x for x in template["agents"] if x["name"]==agent_name][0]
    return agent_di


def get_moderator(template):
    li = [x["name"] for x in template["agents"] if x["moderator"]]
    assert len(li) == 1, "There can be only one ... moderator."
    return li[0]


def instantiate_agents(template, api_key):
    agent_clients = {}
    for agent_data in template["agents"]:
        agent_name = agent_data["name"]
        client = OpenAI(api_key=api_key)
        agent_clients[agent_name] = client
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
        model = "gpt-3.5-turbo"
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(model=model, messages=messages)
    responses = completion.choices
    return responses

def make_responses(client, template, agent_name, prompt, model=None, rounds=1, verbose=True):
    if model is None:
        model = "gpt-3.5-turbo"
    responses = []
    response = respond(client, prompt, model)[0].message.content  # initial response content
    responses.append(response)
    resp_ret = ""
    for i in range(rounds):  # use inner monologue to improve response
        agent_di = get_agent(template, agent_name)
        inner_prompt = ""
        try:
            inner_prompt = agent_di['inner_prompt']
        except:
            inner_prompt = "Given the above prompt and response explain how you can improve the response, and then provide a revised response in the same format as the original response.  Ensure that the revised response does not repeat any points that have already been made and that the name and identity in the initial response does not change.  Ensure that the argument remains consistent and under 100 words."
        mono_prompt = f"{prompt}\nResponse:{response}\n{agent_di['inner_prompt']}"
        messages = [{"role":"user", "content": mono_prompt}]
        completion = client.chat.completions.create(model=model, messages=messages)
        resp = completion.choices[0].message.content
        if verbose:
            print(f"Initial Response:{response}\n")
            print(f"Inner monologue: {resp}\n")
        m = [resp] #re.findall('Action Input: .*$', resp)
        responses.append(m[0])
        response = m[0]
        resp = re.findall("(^.*)", resp)[0]
        resp_ret = resp_ret + f"Round {i+1}: {resp}\n\n"
    return resp_ret, responses

def choose_response(responses, client, prompt, model=None, rounds=1):
    '''
    This function is deprecated and only here for reference purposes
    '''
    '''if model is None:
        model = "gpt-3.5-turbo"
    choose_prompt = f"{prompt} Given the following responses, return the number, and only the number of which response below is the best.\n"
    for i in range(rounds+1):
        choose_prompt+=f"{i}. {responses[i]}\n"
    print(f"\n\nPrompt: {choose_prompt}")
    messages = [{"role":"user", "content": prompt}]
    completion = client.chat.completions.create(model=model, messages=messages)
    print(completion.choices[0].message.content)'''
    return responses[rounds]

config_dir = Path(".")

api_key = os.environ["OPENAI_API_KEY"]
if api_key is None:
    raise ValueError("Add your OPENAI_API_KEY to your environment variables.")
    api_key = st.text_input("Provide your OpenAI API key.")

# Load debate templates and initialize history
templates = load_templates(config_dir)
debates = sorted(templates.keys())

# Select debate
debate = st.selectbox("Select debate:", tuple(debates))
n_rounds = st.select_slider(label="Number of debate rounds", options=range(5,31))
inner_monologue = st.checkbox("Use inner monologue?")
verbose = st.checkbox("Verbose (print in local terminal)")
inner_rounds = st.select_slider(label="Number of inner monologue rounds", options=range(1,6))
template = templates[debate]

# Instantiate agent clients and history
agent_clients = instantiate_agents(template, api_key)
agent_names = list(agent_clients.keys())
if "history" not in locals():
    history = ""
#if round_ not in locals():
#    round_ = 0
moderator = get_moderator(template)
debaters = [x for x in agent_names if x != moderator]

# Set default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.button("Start Debate"):
    #st.session_state.messages.append({"role": "user", "content": prompt})
    for round_ in range(n_rounds):
        if round_ == 0:
            with st.chat_message(moderator):
                message_placeholder = st.empty()
                full_response = ""
                prompt = make_prompt(template, moderator, history)
                responses = respond(agent_clients[moderator], prompt)
                response_content = responses[0].message.content
                chat_response_content = response_content.split("Action Input:")[-1]
                if moderator=="EMH":
                    message_placeholder.markdown("Please state the nature of the moderation emergency.\n\n" + chat_response_content)  # joke for Sam's benefit
                else:
                    message_placeholder.markdown(chat_response_content)
                history = append_to_history(history, response_content)
            st.session_state.messages.append({"role": moderator, "content": chat_response_content})
        # Last round the moderator decides who won
        elif round_ == n_rounds - 1:
            with st.chat_message(moderator):
                message_placeholder = st.empty()
                prompt = make_prompt(template, moderator, history)
                prompt += "\nDecide who won the debate and explain why.  Provide a score of 0-100 for each debater and explain the reason for the score with an itemized break-down, score (0-20), and explanation using the following criteria: Organization and Clarity, Use of Arguments, Use of examples and facts, Use of rebuttal, Presentation Style.  Then give an overall score for each debater."
                #prompt += "\nDecide who won the debate and explain why. Provide a score of 0-100 for each debater and explain the reason for the score with an itemized break-down and explanation of methodology in the format:\n\nSpeaker 1: Criterion 1: score, Criterion 2: score, etc., Overall: score\n\nSpeaker 2: Criterion 1: score, Criterion 2: score, etc., Overall: score."
                responses = respond(agent_clients[moderator], prompt)
                response_content = responses[0].message.content
                chat_response_content = response_content.split("Action Input:")[-1]
                message_placeholder.markdown(chat_response_content)
                history = append_to_history(history, response_content)
            st.session_state.messages.append({"role": moderator, "content": chat_response_content})
        # Debate occurs between two participants otherwise
        else:
            for debater in debaters:
                with st.chat_message(debater):
                    if inner_monologue: 
                        inner_placeholder = st.empty()
                    message_placeholder = st.empty()
                    prompt = make_prompt(template, debater, history)
                    if inner_monologue:
                        inner, responses = make_responses(agent_clients[debater], template, debater, prompt, rounds=inner_rounds, verbose=verbose)  # Create three responses based off of an inner monologue
                        response_content = responses[inner_rounds]  #choose_response(responses, agent_clients[debater], prompt, rounds=inner_rounds)  # Choose the strongest of the three responses
                        inner_placeholder.markdown(f"Inner monologue: {inner}\n\n---\n\n")
                        if verbose:
                            print("---")
                    else:
                        responses = respond(agent_clients[debater], prompt)
                        response_content = responses[0].message.content
                    chat_response_content = response_content.split("Action Input:")[-1]
                    message_placeholder.markdown(chat_response_content)
                    history = append_to_history(history, response_content)
                if inner_monologue:
                    st.session_state.messages.append({"role": f"{debater} Inner Monologue", "content": inner})
                st.session_state.messages.append({"role": debater, "content": chat_response_content})

