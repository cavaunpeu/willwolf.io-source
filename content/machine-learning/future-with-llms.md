Title: Our Future with LLMs
Date: 2023-07-29 10:00
Author: Will Wolf
Lang: en
Slug: future-with-llms
Status: published
Summary: In this post, I explore the evolving world of Language Learning Models (LLMs), considering how they learn, the future of human-LLM conversations, the hallucination problem, compensating data providers, the potential lucrativeness of data annotation, and the idea of a new Marxist struggle.
Image: ../images/future-with-llms/human-mind.png

# What is a Language Model?

To introduce the language model let's talk about human speech. If I were to say, "The boy put on his boots and went to the...," what word comes next? Well, there are many that fit. As an English speaker, you could surely list a few: "store," "park," "bar," even "best" as in "best ice cream spot in town." Conversely, many words could *never* follow this phrase, like "don't," "photosynthesis," or "trigonometry."

And how do you know this? How can you be sure? It's because: you have a "model of language" in your brain, a "language model" that you've acquired over your lifetime. The more language you ingest through interacting with the world the better your model becomes.

# How does ChatGPT know so much?

In the past 30 years, internet users have unwittingly built the largest, broadest, most diverse, most interesting dataset in human history from which to learn machine learning models. These data come in various forms, or "modalities," like images from Instagram, videos from YouTube, audio from various platforms, and text from Wikipedia, Reddit, and many more.

ChatGPT is learned from text. Whereas you’ve trained your language model only on the language you've encountered in the handful of decades for which you've been alive, ChatGPT has been trained on a large chunk of all text ever written on the internet, period. For comparison, [Quora](https://www.quora.com/How-long-would-it-take-you-to-read-the-entire-internet) users estimate that this would take a human roughly "23.8 million years" to "you can't, so it's an impossible question" to "you would be dead long before you even made a start."

This makes it very good at predicting the next word in a phrase, such as our earlier example about the boy and his boots. More generally, it can skillfully continue almost *any* text, such as "Can you recite me the Indonesian national anthem in the style of Shakespeare?" or "What should I make for dinner if I only have salmon and chocolate?" or "What's the best way to get to the moon?"

# How does ChatGPT work?

Creating ChatGPT involves three steps:

1. **Train a language model:** Given a phrase, teach the model how to output, i.e. "predict," the next word. (Repeating this process ad infinitum, appending the predicted word to the end of the initial phrase each time, it can generate a complete response.)
2. **Fine-tune on `(prompt, response)` pairs:** Humans provide both parts of these pairs, giving concrete demonstrations of the tasks the model should accomplish.
3. **Further fine-tune via a model of output quality.** Humans rate the quality of ChatGPT's outputs, a second model learns these relationships, and ChatGPT learns to output high-quality responses via this second model. This process is known as ["Reinforcement Learning from Human Feedback"](https://huggingface.co/blog/rlhf) (RLHF).

# This is a story about data

Throughout my career in machine learning, I've learned almost every ML problem is a story about data. Where does it come from? What's wrong with it? How much does it cost? How do we de-bias it? How do we get more of it? Which data should we label next? And on. ChatGPT is no different.

With this in mind, here are a few keys points regarding where we stand today:

**Static knowledge**: The language model has been trained on a large chunk of the written internet, dated through September 2021 (as users will now know by heart, as the model loves to restate this limitation). Encapsulated in this data is the knowledge required to solve a substantial number of *static-knowledge* tasks. For example, the model can summarize news articles; as the nature of summarization doesn't really evolve year over year, the model does not per se require any more data to accomplish this task. It has enough already.

**Dynamic knowledge**: Conversely, imagine that we'd like to translate classical Greek into modern English. Well, what does "modern" mean? Language constantly [evolves](https://www.amazon.com/Because-Internet-Understanding-Rules-Language/dp/0735210934) to include new vocabularly and modes of expression. So, while the fundamentals of translation don't really change, the contemporary details do. To keep pace with these details, the model needs to be updated with examples of this text. I surely wasn't saying "that's lit" myself ten years ago.

**Novel knowledge**: Finally, novel knowledge defines the set of tasks or abilities that the model has never encountered. For instance, a novel discovery in physics, e.g. room-temperature [superconductivity](https://phys.org/news/2023-07-korean-team-room-temperature-ambient-pressure-superconductor.html) is an example of *dynamic knowledge* if this work is an *extension* of the scientific knowledge, logical reasoning, historical expectations, etc. that the model already posseses. Conversely, this discovery is an example of *novel knowledge* if it is predominantly composed of never-before-seen ways of conceptualizing the world, e.g. "a new mathematics," alien anatomy, etc.

The vast majority of knowledge is either static or dynamic. However, for completeness, we leave a small sliver of space for novel knowledge as well.

**Human annotators**: Human annotators (paid and trained by OpenAI) have provided the data required for the supervised fine-tuning and RLHF [steps]({filename}/machine-learning/future-with-llms.md). Should we wish to expand the "foundational" set of tasks that we explicitly want the model to solve, or update our "preference" regarding the way in which the model expresses itself, we'll need more annotations.

# A menu of questions

In this post, I explore our future with LLMs from the perspective of *data*. I'll do so by asking and answering a series of questions—touching on the methods, the players, the economics, and the power struggles that potentially lie ahead.

1. How will LLMs learn new information?
2. What will we do with human-LLM conversations?
3. How do we solve the hallucination problem?
4. How will we compensate data providers?
5. Will data annotation be lucrative?
6. Is this the new Marxist struggle?

Let's begin.

## How will LLMs learn new information?

I work as a software engineer. If you don't, you might believe that all engineers have committed to memory the knowledge and syntax required to solve every task we ever encounter throughout our day. Alas, we haven't. Instead, we commonly use "question-answer" sites like [Stack Overflow](https://stackoverflow.com/) to see how other developers have solved the problem at hand. Before ChatGPT, I used StackOverflow almost daily.

Several sources [posit](https://www.similarweb.com/blog/insights/ai-news/stack-overflow-chatgpt/) that ChatGPT is directly cannibalizing Stack Overflow traffic. Personally, I don't find the statistics they provide particularly convincing. So, let's use an anecdote instead: since I started using ChatGPT in ~4 months ago, I have not been on Stack Overflow once. Why wait for human responses when ChatGPT responds instantly?

If other developers are doing the same, Stack Overflow "freezes." In other words, no new human programming knowledge is published at all. Like language translation, coding implies translating a set of logical expressions in ones brain into machine-readable instructions. Across the spectrum of present and future programming languages, the fundamentals generally don't change. However, the details do. In this vein, being a software engineer (as a human, or an AI model) is a dynamic-knowledge task. With the next programming language to hit the market, how will developers know how to use it?

Let's consider [Mojo](https://www.modular.com/mojo), a new programming language built specifically for AI developers. As Mojo is a superset of Python, our knowledge about Python will still apply. However, Mojo will bring new features that we haven't seen before. Simply put, how will ChatGPT learn how to read and debug Mojo?

For reading, the answer might be simple: include the Mojo [documentation](https://docs.modular.com/mojo/) in the model's training set. This provides a basis for understanding the syntax and semantics of Mojo.

For debugging, I think it's more subtle. [GitHub Copilot X](https://github.com/features/preview/copilot-x)—the LLM-powered tool that helps you write and debug code—will now capture and send your terminal context back to OpenAI itself. As such, with this beta, the LLM is actively "acquiring new data" regarding the workflows, questions, patterns, etc. inherent in programming in Mojo. (And furthermore, as these humans contribute the data that improves the model, they're still paying $20/month for the use of the tool itself!)

Taken together, for the model to update its understanding of dynamic and novel knowledge tasks, we must provide it with new data. In what follows, we codify the nature of this provision along three main axes: implicit vs. explicit, fidelity, and velocity.

### Implicit vs. explicit

The nature of data provision will *range* from implicit to explicit. Capturing the Mojo developer's terminal context is an example of implicit data provision. Curating model training examples about how to resolve common Mojo errors is an example of explicit data provision. Answering empirical questions on Stack Overflow has elements of both.

Broadly, implicit data will be easier to collect (there's more of it to go around), and vice versa.

### Fidelity

In the three cases just mentioned, we assume humans act *rationally*, meaning that they earnestly try to produce the "right information" in order to solve their problem, which we can ideally add to the model's training set. However, this data "fidelity" varies across each case—as a function of who is providing the data, and what their incentives and requirements are.

In the "capturing terminal context" setting—implemented naively—we are capturing information from *all* developers. Some might be good, others bad. While most developers are likely "trying to solve their problem," or "debugging code until it works," the quality, and therefore fidelity, of this information will vary as a function of their skills.

In Stack Overflow, the same "available to all" feature applies. However, there is both an additional social pressure placed on users of the site to provide correct information—people don't want to look silly in front of their peers—as well as an explicit feedback mechanism—answers deemed correct get "upvoted," and vice versa. Nominally, these constraints increase data fidelity.

Finally, we assume that the "manually curate training set examples" setting gives the highest-fidelity data of the three. Why? A company is paying a human to explicitly teach the model information. Before annotation, they ensure the human has the right qualifications; after annotation, they likely review the results. Taken together, the highest scrutiny gives the highest fidelity.

### Velocity

Finally, across the three settings, the *speed* with which we can generate a large and diverse dataset is different. In manual curation, it's slowest (single human). On Stack Overflow (many human-human interactions), it's faster. In the terminal (many human-machine interactions), it's fastest (and probably by a lot). A "many machine-machine interactions" setup, e.g. copies of a reinforcement learning system where an AI plays the part of the programmer, gets feedback from the language, and iterates, all running in parallel, would be even faster...

### So where do I get data?

With the above in mind, companies will seek out data sources that make an optimal tradeoff between: the pragmatic implications of collecting implicit vs. explicit data, the fidelity of the data provided, and the speed with which it is generated. Broadly, implicit data will be easier to collect, low fidelity, and high velocity. Explicit data will be harder to collect, high fidelity, and low velocity.

**Overall, companies will need to identify the "data provision venue" that makes the right tradeoffs for them and their model.** Then, they'll need to be strategic about how to "drop their net" into this stream and catch the passing fish.

## What will we do with human-LLM conversations?

ChatGPT already boasts a staggering [100 million users](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/). An enormous quantity of human-LLM conversations are taking place daily. What do we do with these interactions? What valuable data can be gleaned from these conversations, and how can OpenAI use this information?

### Product discovery

In one approach, we could summarize the conversations, embed the summaries, cluster them, then use the cluster's centroid to generate a cluster label. In this way, we can start to understand what people are using the model for. Furthermore, ChatGPT could be trained to proactively ask users for feedback or suggestions, gauging their interest in potential new products.

### Reinforcement learning

Another approach might involve using conversation outcomes as learning signals for reinforcement learning. For instance, annotators could label successful and unsuccessful conversations, and these labels could be used as rewards in reinforcement learning algorithms like RLHF. This feedback could also be supplemented by binary labels generated by the model itself.

### Learning optimal prompts

Lastly, by associating tasks with clusters of conversations and their descriptions, the system could learn to generate optimal prompts for those tasks. A hypothetical workflow might involve generating a prompt, having two LLMs engage in a conversation based on this prompt, scoring the result using our reward signal, and updating the prompt-generation policy using reinforcement learning methods like Proximal Policy Optimization (PPO), similar to what's used in the RLHF step of training ChatGPT.

**Taken together, companies will use human-LLM conversations to**:

- Discover and prioritize novel applications or products by analyzing usage patterns and directly soliciting user feedback.
- Improve the model itself, based on conversation outcomes and reward signals.
- Enhance the user experience by understanding and helping users accomplish their existing goals more effectively.

## How do we solve the hallucination problem?

"Hallucation" is when an LLM says things that have no basis in fact or reality. If we knew *when* the model did this, we could simply restrict those outputs. If we knew *why*, we could likely design better models that hallucinate less. Unfortunately, the answers to these [questions](https://cims.nyu.edu/~sbowman/eightthings.pdf) remain elusive.

[Retrieval](https://en.wikipedia.org/wiki/Information_retrieval#:~:text=Information%20retrieval%20is%20the%20science,of%20texts%2C%20images%20or%20sounds.) models select outputs from a fixed "menu" of choices. In this way, we implicitly "solve" the hallucination problem by explicitly restricting a priori what the model can and can't "say." Generative models, on the other hand, make a different tradeoff: allowing the model to generate novel content *ex nihilo* implies forfeiting some control over what it can say.

Paying rational human annotators to "correct" all recorded hallucinations would likely improve this situation. However, much like dictatorship, policing the actions and behaviors of every constituent is not a scalable strategy. In addition, the question of *who* decides what "correct" actually means remains open for virulent debate. In the context of ChatGPT, this is OpenAI. Similarly, in the context of the 2020 presidential election, it was Facebook that decided what content was and was not acceptable to promote. Combining the two, an interesting question arises: How do we solve the hallucination problem without a centralized authority? Said differently, how do we build models whose voice represents that of the broader consensus? It is extremely likely that a discussion surrounding some form of this question will unfold in the coming years.

My technical background is largely in machine learning. However, I've been working in crypto for the past two years. In this section, I'll borrow an idea from the latter and apply it to the former. The following idea may be fancfiful and impractical, and is certainly not the only way to approach this problem. Nonetheless, it makes for an interesting thought experiment.

### Proof of Stake

The crypto world has spent the last ~15 years trying to answer a similar question: How do we build a scalable, trustworthy system for the transfer of monetary value that does not rely on centralized intermediaries? To date, one of the key mechanisms used to achieve this end is Proof of Stake (PoS). PoS is a consensus algorithm where partipicants, or "validators," are collectively entrusted to verify the legitimacy of transactions. To incentivize earnest behavior, PoS employs the following mechanism:

- Participants are paid to validate transactions.
- Prior to validating transactions, participants "stake" capital. This "stake" is like a hold placed on your credit card when renting a car.
- The more capital you "stake," the more likely you are to be selected to validate transactions.
- If other partipicants deem your behavior dishonest, your "stake" is taken (and you do not get paid).

Take together, PoS promotes transaction fidelity with economic incentives and penalties. Dishonest participants may lose their staked tokens, creating a self-regulating system where everyone has a vested interest in success.

### Applying Proof of Stake to LLMs

How might we apply PoS to LLMs? In effect, the users are the validators who ensure the legitimacy not of transactions, but of model outputs. Validators would have a stake, such as points, reputation, privileges, or digital currency, at risk, creating a vested interest in the accuracy of feedback. Then, model outputs would be periodically selected for collective review. Validators would be rewarded for proposing valid feedback in consensus with other users. Conversely, those providing inaccurate feedback or acting maliciously would lose their stake.

Much like PoS in the blockchain world, this system is not without its challenges. For instance:

- How do we ensure that a small number of "high-stake" users don't control the system?
- Will the collective expertise of users empirically "satisfy" the model provider? Elon, maybe; Sam ([Altman](https://en.wikipedia.org/wiki/Sam_Altman)), unclear.
- What balance between rewards and penalties promotes truthful feedback without stifling participation?
- Etc.

Finally, the question of *when or why* this type of system might ever be introduced is ultimately a question of who retains power. Will LLMs simply be services offered by private companies? Will governments mandate their use as a public utility informed by, and built for, the body politic itself? I don't have the answers to any of these questions. However, my popcorn is ready!

**Overall, Proof of Stake is but one approach to solving the hallucination problem.** As an algorithm for decentralized consensus, its relevance will correlate with the evolving narratives of scalability, fairness, and the distribution of power in the context of LLMs.

## How will we compensate data providers?

Much like LLMs, business is a story about data as well. For instance, historical customer purchase data allows a business to prioritize which products and services to sell in the the future. Customer characteristics like age, location, gender, and political preferences enable more targeted advertisements. Data collected in feedback forms hint at new products that customers might buy.

Before the internet, companies employed a variety of traditional methods to collect these data: questionnaires or comment cards, tracking purchases and preferences through loyalty cards, and utilizing demographic information and other publicly available data. With the internet, the game changed overnight. Now, companies could track every page visit, click, and even [eye movements](https://theconversation.com/companies-are-increasingly-tracking-eye-movements-but-is-it-ethical-191842#:~:text=Apple%2C%20Google%2C%20Snap%2C%20Microsoft,contact%20lenses%20and%20AR%20headsets.). In addition, they encouraged the creation of data expressing *implicit* user preferences, such as pictures on Instagram, chats between friends on Messenger, videos on YouTube. Much like before, these data offered clear value to businesses.

Other types of data were valuable as well. Expert replies on question-answer sites like Quora offered informational value to users, and garnered reputational value for the author herself. Basic data annotation enabled machine learning practitioners to train models. In [this](https://open.spotify.com/episode/26xt8MwmfaBlmU6GFjYumu) episode of The Daily, Sheera Frenkel discusses how fantasy story writing communities a sense of purpose and satisfaction to writers, and of course entertainment value to readers. Finally, online courses on sites like [Platzi](https://platzi.com/) offer clear educational value to students.

Overall, these data remain valuable to diverse parties in myriad ways.

### Current compensation for individuals

In exchange for the data above, its *creators* are *compensated* through the following three means:

- **Financial**: Direct payments, royalties, or subscription fees for music, art, literature, educational material, data annotation, and more.

- **Reputation**: On platforms like Quora or Stack Overflow, individuals gain recognition based on the quality of their responses, knowledge, and expertise, enhancing their personal brand within a community.

- **Moral**: Individuals derive personal satisifaction from contributing something unique to the world.

### What do LLMs providers value?

LLMs represent an entirely new consumer of data. As such, as a function of the ways in which these models will be used, LLM providers value different things. Already, "beyond its mastery of language, GPT-4 [the LLM behind ChatGPT] can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance." Conversely, beyond hallucination, what are these systems *not* good at? What do they need to be maximally useful and achieve widespread user adoption?

- mojo-like updates. could compensate with royalty payments, via provenance proving data structures.
- multi-turn feedback. could compensate with just free usage of the LLMs themselves.
- authoritative answers; it can't just be "this is the official OpenAI answer," right? we want: "this is an official answer from this person of this perspective." as to how these answers are solicited, i'm not sure; could be another social mechanism design task. however, reputational benefit awaits the person who is cited (and they'll be *the* answer from *the* LLM). when you shrink the podium, its space becomes all the more valuable.
- what kind of ~spiritual compensation will come to those who just... talk to LLMs? seems like it could suck? will LLM providers explicitly make models say things that "make people feel better?" will they "soma" them to sleep?

## Will data annotation be lucrative?

From the humble beginnings of 'cat' or 'dog' labels to resolving sophisticated moral dilemmas for autonomous vehicles, the role of **data annotators** has evolved tremendously alongside the advances in machine learning. In our journey towards a future shaped by Large Language Models (LLMs), their contribution continues to be vital, and arguably, more complex and specialized than ever.

**The Evolution of Data Annotation:**
Historically, the role of data annotators was rather straightforward, confined to supervised learning tasks. However, as machine learning tasks grew more intricate, the skill requirements for data annotators escalated as well. Training these individuals became a more significant burden for task creators, dealing with tasks such as [semantic role labeling](https://en.wikipedia.org/wiki/Semantic_role_labeling).

**The LLM Shift:**
In the world of LLMs, data annotators' roles have expanded beyond mere labeling. Today, they guide model behavior through Reinforcement Learning with Human Feedback (RLHF), influencing the models by embedding human preferences into them. These annotators hold the keys to the behavior guidelines of LLMs and, in essence, shape the future of machine learning.

**What Lies Ahead:**
As we steer into the future, the tasks of data annotators will likely grow more nuanced and specialized. A hypothetical scenario in this "logical extreme" could see data annotators as trained ethicists, making high-stakes decisions akin to resolving moral dilemmas for autonomous cars. Does this mean that the role of data annotators will become more lucrative? Quite possibly. Yet, the question of who gets to appoint these annotators and how this process is regulated will significantly influence this trajectory.

Will these models remain in the hands of private companies, leading to fewer but more specialized annotators? Or will we see a shift towards a more public, democratic process where lawmakers or public opinion shape the annotation landscape?

**Politics and Data Annotation:**
Intriguingly, the future of data annotation could be seen as a battle between a highly educated autocracy and a democracy. In an autocratic model, specialized and "intelligent" individuals make decisions, not unlike countries requiring public officials to pass competency exams, as seen in [South Korea](https://en.wikipedia.org/wiki/Republic_of_Korea_public_service_examinations). In contrast, a democratic model invites broader public participation, embodying a more bottom-up approach.

**In summary:**
In the end, the future of data annotation — its evolution and potential lucrativeness — hangs in the balance of legal frameworks, societal choices, and the ever-increasing sophistication of the questions these models need to answer. As we lean into this future, the complexity and importance of the data annotators' roles only seem to magnify. Just another day in the exhilarating roller coaster ride of machine learning and artificial intelligence!

## Is this the new Marxist struggle?

**Model Providers and Data Providers:**
In the context of Marxist theory, [OpenAI] and similar entities have become the bourgeoisie of our time, the owners of the means of production. They control AI models, which in today's digital age, represent the critical tools of production. On the flip side, data providers serve as the proletariat. Their labor, in the form of data, powers these AI models in the same way manual labor fuels traditional industries.

**Precedence of the Struggle:**
However, this struggle for data remuneration isn't a new development. It has precursors in recent history, particularly with the rise of social media platforms like Facebook and Google in the past 20 years. These tech juggernauts harvested user data without offering direct compensation, implicitly trading their free services for user data. This arrangement sparked debates around power dynamics and financial implications. But a significant number of users, deriving value from these platforms, remained largely unperturbed by the absence of explicit compensation.

**Increasing Applicability of AI:**
The advent of Large Language Models (LLMs), however, introduces a crucial difference: the dramatically expanding range of AI applications. The services promised by LLMs aren't confined to social communication or productivity enhancement. Instead, they're projected to fill roles as diverse as tax consultants, therapists, fitness coaches, educators, and even advocates for refunds. As AI permeates deeper into our daily lives, the issues of labor and compensation in data provision become more stark and salient.

**Compensation Concerns:**
The growing discontent over the lack of compensation for data provision may correlate directly with whether these AI services are enhancing users' lives or supplanting their jobs. In contrast to Facebook or Google, which mainly served as communication and information tools, LLMs have the potential to replace entire job roles, like accounting. The threat of job displacement due to AI could accelerate conversations around compensation, fueling a struggle akin to labor dynamics.

**Implications for Job Loss and Economic Value:**
Furthermore, we should consider potential dystopian futures, as depicted in Yuval Noah Harari's "Homo Deus". In this extreme scenario, widespread automation could lead to substantial job losses, leaving people with only their data as a means of economic contribution. Such a situation would intensify the "Marxist struggle" around data and compensation, underscoring the importance of these issues in societal discourse.

**In summary:**
Whether we're on the precipice of a new Marxist struggle over data and compensation is an intricate question interwoven with societal, economic, and technological factors. But one thing is clear: as AI continues to pervade our lives, these debates will grow louder and more urgent. It's our collective responsibility to engage in these conversations, shaping a future where technological advancement and societal welfare can coexist harmoniously.

# Conclusion

The LLM story is sure to evolve quickly. It's unclear where it will go. In the best case, human knowledge, satisfaction, and general well-being compound exponentially. Fingers crossed.

# References

```
[^1]: @online{johnson2023,
    title={Exploring Methods for Model Selection and Training Data Curation},
    author={Kathryn Johnson and others},
    year={2023},
    eprint={2210.11610},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    note={\url{https://arxiv.org/pdf/2210.11610.pdf}}
}
[^2]: @online{wired2023,
    title={Stack Overflow Will Charge AI Giants for Training Data},
    year={2023},
    note={\url{https://www.wired.com/story/stack-overflow-will-charge-ai-giants-for-training-data/}}
}
[^3]: @online{analyticsindiamag2023,
    title={Is This the Beginning of the End of Stack Overflow?},
    year={2023},
    note={\url{https://analyticsindiamag.com/is-this-the-beginning-of-the-end-of-stack-overflow/}}
}
[^4]: @online{stackoverflow2023,
    title={Community is the Future of AI},
    year={2023},
    note={\url{https://stackoverflow.blog/2023/04/17/community-is-the-future-of-ai/}}
}
[^5]: @online{zhang2023,
    title={On the Community and Automation of Machine Learning},
    author={Jiehang Zhang and others},
    year={2023},
    eprint={2306.15774},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    note={\url{https://arxiv.org/pdf/2306.15774.pdf}}
}
[^6]: @online{lin2023,
    title={Harnessing Data from the Wisdom of Crowds for Machine Learning},
    author={Simon Lin and others},
    year={2023},
    eprint={2306.08302},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    note={\url{https://arxiv.org/abs/2306.08302}}
}
[^7]: @online{singh2023,
    title={Lessons Learned from the Deployment of an AI-Based Teacher Assistant},
    author={Harshit Singh and others},
    year={2023},
    eprint={2305.18339},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    note={\url{https://arxiv.org/abs/2305.18339}}
}
[^8]: @online{sun2022,
    title={The High Stakes of Low-Level Details: A Close Reading of Disturbing Behaviors in AI Chatbots},
    author={Sun, J. and others},
    year={2022},
    eprint={2202.05144},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    note={\url{https://arxiv.org/abs/2202.05144}}
}
[^9]: @online{rosenfeld2021,
    title={Deciphering the components of national AI strategies},
    author={Rosenfeld, E. and Vincent, N.},
    year={2021},
    eprint={2108.13487},
    archivePrefix={arXiv},
    primaryClass={cs.CY},
    note={\url{https://arxiv.org/abs/2108.13487}}
}
[^10]: @online{zhao2023,
    title={Ethics as a Service: a pragmatic operationalisation of AI Ethics},
    author={Zhao, J. and others},
    year={2023},
    eprint={2307.10169},
    archivePrefix={arXiv},
    primaryClass={cs.CY},
    note={\url{https://arxiv.org/pdf/2307.10169.pdf}}
}
[^11]: @online{frase2022,
    title={End User Licensing Agreements for AI Systems},
    author={Frase, L. and others},
    year={2022},
    eprint={2212.10450},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    note={\url{https://arxiv.org/abs/2212.10450}}
}
[^12]: @online{deng2023,
    title={The Impact of Stack Overflow Data on AI Model Performance},
    author={Deng, J. and others},
    year={2023},
    eprint={2306.11644},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    note={\url{https://arxiv.org/abs/2306.11644}}
}
[^13]: @inproceedings{findings-emnlp2021,
    title={A Review of the Evolution of Ethical Guidelines for AI Development},
    author={N/A},
    booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (Findings)},
    year={2021},
    note={\url{https://aclanthology.org/2021.findings-emnlp.192.pdf}}
}
[^14]: @online{liu2023,
    title={A Sociotechnical Framework for the Development and Deployment of AI},
    author={Liu, C. and others},
    year={2023},
    eprint={2302.13007},
    archivePrefix={arXiv},
    primaryClass={cs.CY},
    note={\url{https://arxiv.org/pdf/2302.13007.pdf}}
}
[^15]: @book{lanier2013,
    title={Who Owns the Future?},
    author={Jaron Lanier},
    publisher={Simon & Schuster},
    year={2013},
    chapter={Chapter 3}
}
[^16]: @online{openai2020,
    title={ChatGPT},
    year={2020},
    note={\url{https://openai.com/blog/chatgpt}}
}
[^17]: @online{openai2021,
    title={Instruction Following},
    year={2021},
    note={\url{https://openai.com/research/instruction-following}}
}
[^18]: @online{openai2021,
    title={Learning from Human Preferences},
    year={2021},
    note={\url{https://openai.com/research/learning-from-human-preferences}}
}
[^19]: @online{openai2021,
    title={OpenAI Baselines: PPO},
    year={2021},
    note={\url{https://openai.com/research/openai-baselines-ppo}}
}
[^20]: @online{schulman2017,
    title={Proximal Policy Optimization Algorithms},
    author={Schulman, J. and others},
    year={2017},
    eprint={1707.06347},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    note={\url{https://arxiv.org/pdf/1707.06347.pdf}}
}
[^21]: @online{you2023,
    title={A Survey of Annotation Methods for AI Model Development},
    author={You, Z. and others},
    year={2023},
    eprint={2304.01852},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    note={\url{https://arxiv.org/pdf/2304.01852.pdf}}
}
[^22]: @online{lee2022,
    title={Public Engagement in AI Development: A Case Study},
    author={Lee, D. and others},
    year={2022},
    eprint={2209.01538},
    archivePrefix={arXiv},
    primaryClass={cs.CY},
    note={\url{https://arxiv.org/abs/2209.01538}}
}
[^23]: @online{huggingface2022,
    title={Training Chatbots with Reinforcement Learning at Hugging Face},
    year={2022},
    note={\url{https://huggingface.co/blog/rlhf}}
}
[^24]: @online{durmonski2020,
    title={Who Owns The Future: 5 Things We Learned From Jaron Lanier},
    year={2020},
    note={\url{https://durmonski.com/book-summaries/who-owns-the-future/#5-lesson-2-ordinary-people-are-not-compensated-for-the-information-taken-from-them}}
}
https://cims.nyu.edu/~sbowman/eightthings.pdf
```