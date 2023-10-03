# Related Work

### Liang et al. (2023): Encouraging divergent thinking in large language models through multi-agent debate
* [Link](https://arxiv.org/pdf/2305.19118.pdf)
* [Code](https://github.com/Skytliang/Multi-Agents-Debate)
* Proposes a “multi-agent debate (MAD), in which multiple agents express their arguments in the state of 'tit for tat' and a judge manages the debate process to obtain a final solution”.
* Applied to work through arithmetic and machine translation problems.
* "LLMs might not be a fair judge if different LLMs are used for agents".
* Has a "devil" represent the affirmative and the "angel" represent the negative. The angel corrects the devil and a judge decides which side is right.
* Aside: This makes me wonder if our approach could become a sort of [dialectical method](https://en.wikipedia.org/wiki/Dialectic), where one agent makes a proposition (thesis), another agent counter-propositions (antithesis), and the judge / arbiter synthesizes a takeaway.

### Kai et al. (2023): Examining the inter-consistency of large language models: An in-depth analysis via debate
* [Link](https://ui.adsabs.harvard.edu/abs/2023arXiv230511595X/abstract)
* Proposes "Formal Debate framework (FORD)", which features "a three-stage debate aligned with real-world scenarios: fair debate, mismatched debate, and roundtable debate."
*	"Stronger LLMs tend to dominate debates by adhering to their perspectives, while weaker ones are more likely to change viewpoints".
*	Important to have "a competent judge".
*	Aside: Our contribution could be the proposal of (another, yet novel!) debate framework. Also, it may be interesting to see if agents that have a lower intelligence (via integrative complexity or by virtue of the sophistication of LLM) are more easily persuaded. And then, what kinds of arguments tend to be most persuasive for simpler agents.

### Du et al. (2023): Improving factuality and reasoning in language models through multiagent debate
*	[Link](https://arxiv.org/pdf/2305.14325.pdf)
*	[Code](https://composable-models.github.io/llm_debate/)
*	Uses debate among agents to improve LLM responses.
*	"Our findings indicate that this approach significantly enhances mathematical and strategic reasoning across a number of tasks"
*	Paper claims method reduces factual inaccuracies.
*	"Findings suggest that such ‘society of minds’ approach has the potential to significantly advance the capabilities of LLMs".
*	Uses the response of the first agent to inform the response of the second, and so on, to iteratively improve response quality.
*	"Found prompts that encouraged models to be more ‘stubborn’ based on their own solutions led to longer debates and a better final solution".
*	Used approach to solve arithmetic, mathematical reasoning, and chess move prediction tasks.
*	Metrics: Nature of tasks made it easy to evaluate performance.
*	Aside: How do we quantify performance?

### Chan et al. (2023): ChatEval: Towards better LLM-based evaluators through multi-agent debate
*	[Link](https://arxiv.org/pdf/2308.07201.pdf)
*	[Code](https://github.com/chanchimin/ChatEval)
*	Presents ChatEval, a “multi-agent referee team” to “autonomously discuss and evaluate the quality of generated responses from different models on open-ended questions and traditional natural language generation (NLG) tasks".
*	Claims to "find that the diverse role prompts (different personas) are essential in the multi-agent debate process; that is, utilizing the same role description in the prompt can lead to a degradation in performance".
