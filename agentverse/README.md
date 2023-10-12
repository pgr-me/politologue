# Description
- examples of AgentVerse multi-agent environment
- complete set up required for AgentVerse CLI interaction

## Notebook Use 
- Upload to Google Colab
- Drop relevant config in the left directory
- Run

## Docker Use
### Pull the Image
```
docker pull oreb/jhu_llm:1.0
```

### Run an AgentVerse Debate Simulation
```
# run the container
> docker run -d -e "OPENAI_API_KEY=<api key here>" oreb/jhu_llm:1.0

# get the container name
> docker ps

# exec into the container
> docker exec -it <container_name> /bin/bash

# run agentverse debate simulation
> python3 main_simulation_cli.py --task simulation/debate_3players

# see the output!

# when you're done
> exit
```

### Update the Prompts or Personas in the Config
```
# replace the config
> docker cp /path/to/new/file/on/host <container_name>:/agentverse/task/simulation/debate_3players/.config

# exec into the container
> docker exec -it <container_name> /bin/bash

# run agentverse debate simulation
> python3 main_simulation_cli.py --task simulation/debate_3players

# see the output!
```

## Updating the AgentVerse Config
![agentverse_config](https://github.com/pgr-me/politologue/assets/32425398/9b2d5012-3559-49fd-aa5f-02716fabc5dd)

## Notebook Examples
- `3player_debate.ipynb`: 
    - 3 player debate between Cato and Caesar, moderated by Cicero.
    - requires OpenAI API key, which will be prompted for
    - LLM: `GPT-4`
    - config: `3player_debate.yaml`

## TODO:
- 3 player group discussion setup. There's a specific environment for group discussions that I want to configure.
- Setup GUI. I think this can be hosted somewhere; seemingly nice interface for viewing and augmenting the dialogues.

