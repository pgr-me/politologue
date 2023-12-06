# Proposal

## Abstract

Add.

## Introduction

Add.

## Related Work

In this section, we summarize other papers that focused on using multiple agents to debate to improve response quality.

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

## Approach

### Specification of Political Personas

In this section, we list three historical political debates in which each debate has two opposing views embodied in a persona. We define each debate and its personas in the following subsections.

#### Roman Republicanism vs. Imperialism

##### 



**Republicanism** Marcus Porcius Cato was a Roman politician and staunch opponent of Caesar.  He championed traditional Roman values, including and especially representative government.  Through his career, Cato defended the authority of the Roman Senate against Caesar and his popular policies because he felt that these reforms were destabilizing the bedrock of Roman society and political order.  When Caesar triumphed in the Roman Civil War, Cato killed himself rather than submit to Caesar's autocracy.  In geneal, Cato held staunch opposition to the very idea of autocracy and tyranny, no matter how well intentioned.

**Imperialism** A patrician by birth, Julius Caesar became a Roman general and statesman.  Through his military career, Caesar had several successes, including the conquest of Gaul and the invasion of Britain leading to gaining power and prestige within the Roman Republic.  He began to consolidate power in the First Triumvirate, but this alliance broke down.  At this point, Caesar crossed the Rubicon and began the Roman Civil War.  Caesar emerged victorious and declared himself dictator.  He led many political reforms which allowed him to consolidate more power to become dictator for life, the Emperor.

##### Greek Direct Democracy vs. Anti-Democracy

**Direct Democracy** Ephialtes was an ancient Athenian politician who is primarily known for his role in the political reforms of Athens during the early stages of the Peloponnesian War.  Ephialtes was a proponent of radical democratic changes and sought to reduce the power of the aristocracy and strengthen the democratic system by shifting more power towards the Assembly, which was comprised of all male citizens, regardless of wealth or social status. Ephialtes initiated measures that paved the way for a more direct form of democracy. This included reducing the powers of the Council of the Areopagus, thereby enabling the Assembly to have a greater say in the decision-making process. Ephialtes' reforms ultimately contributed to the development of a more inclusive and egalitarian democratic system in Athens.

**Anti-Democracy** Plato was an ancient Greek philosopher and student of Socrates. He founded the Academy in Athens, one of the earliest known institutions of higher learning in the Western world. One of his most notable ideas was the concept of the Philosopher King, which is in direct opposition to the prevailing Athenian belief in democracy.  Plato believed that the ideal state should be ruled by philosopher-kings, individuals with a deep love for wisdom and a profound understanding of truth and justice. He argued that these enlightened rulers, possessing a unique form of knowledge, would make decisions based on reason and the pursuit of the highest good, rather than personal gain or popular opinion, thus creating a just and harmonious society.  Conversely, Plato believed that direct democracy was a dangerous form of government in which the basest of personalities would triumph.

#### American Patriots vs. Loyalists

**Patriots** John Adams, one of the Founding Fathers of the United States of America, was an early advocate for American independence from Britain.  As a delegate to both Continental Congresses, Adams was a prime mover for the Declaration of Independence in 1776. He believed strongly that the American colonies should break free of Britain's unfair taxation policies and other restrictions on their liberty, and he felt the King and Parliament were tyrannical and that the colonies deserved their own self-government.  Adams was also instrumental in persuading Congress to declare independence and assisted Thomas Jefferson in drafting the Declaration of Independence.

**Loyalists** William Franklin was the son of American Founding Father Benjamin Franklin and the governor of New Jersey after being appointed by King George III.  As tensions rose between Britain and the American colonies, Franklin opposed independence and defended the authority of the crown.  He believed Britain had the right to tax the colonies and thought American rights could be protected while remaining part of the Empire. Franklin tried to keep New Jersey neutral in the Revolutionary War but ultimately was arrested for supporting Loyalists.  Even after the war, Franklin continued advocating the Loyalist position while living in Britain.

## Results

Add.

## Discussion

Add.

## Conclusion
