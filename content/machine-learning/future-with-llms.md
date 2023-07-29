Title: Our Future with LLMs
Date: 2023-07-29 10:00
Author: Will Wolf
Lang: en
Slug: future-with-llms
Status: published
Summary: In this post, I explore the evolving world of Language Learning Models (LLMs), considering how they learn, the future of human-LLM conversations, the hallucination problem, compensating data providers, the potential lucrativeness of data annotation, and the idea of a new Marxist struggle.
Image: ../images/future-with-llms/human-mind.png

# What is a Language Model?

A language model is best understood through human speech. If I were to say, "The boy put on his boots and went to the...," what words might come next? You could probably list a few sensible options: "store," "park," "bar," even "best" as in "best ice cream spot in town." On the other hand, there are words that would almost certainly never follow this phrase, like "don't," "photosynthesis," or "trigonometry."

How do you know this? You have a "model of language" in your brain, a "language model" that you've acquired over your lifetime. The more language you ingest, the better your model becomes.

# How does ChatGPT know so much?

Over the past 30 years, internet users have unwittingly built the largest, most robust, most diverse, most interesting dataset in human history for learning machine learning models. These data come in various forms, or "modalities," including images from Instagram, videos from YouTube, audio from various platforms, and text from Wikipedia, Reddit, and many other sources.

ChatGPT, a "large language model" or LLM, is trained on a meaningful portion of all text data online. This makes it very good at predicting the next word in a phrase, such as our earlier example about the boy and his boots.

# How does ChatGPT work?

Creating ChatGPT involves three steps:

1. **Train a language model:** Given a phrase, teach the model how to predict the next word. (It can then append this next word to the phrase, and repeat this process ad infinitum to generate a full sentence.    )
2. **Fine-tune on `(prompt, response)` pairs:** Humans provide both parts of these pairs, giving concrete demonstrations of the tasks the model should accomplish.
3. **Further fine-tune via a model of output quality.** Humans train a model to assign a quality score to a given output; ChatGPT is then refined to only output high-quality responses. This process is known as ["Reinforcement Learning from Human Feedback"](https://huggingface.co/blog/rlhf) (RLHF).

# This is a story about data

Throughout my career in machine learning, I've learned that almost ML problem is a story about data. Where does it come from? What's wrong with it? How much does it cost? How do we get more of it? And on and on.

ChatGPT is no different. Its language model has been trained on a huge chunk of the internet, and it's been fine-tuned using data provided by human annotators. The extent of its abilities, or the ways in which it could be improved, are determined by the nature and amount of the data it has been trained on.

# A menu of questions

In this post, I explore our future with LLMs from the perspective of *data*. I'll do so by asking and answering a series of questions, touching on the methods, the players, the economics, and the power struggles that potentially lie ahead.

1. How will LLMs learn new information?
2. What will we do with human-LLM conversations?
3. How do we solve the hallucination problem?
4. How will we compensate data providers?
5. Will data annotation be lucrative?
6. Is this the new Marxist struggle?

## How will LLMs learn new information?

Imagine you're a software engineer. You might think that the job entails memorizing all code required to solve the task. However, most engineers rely heavily on "question-answer" sites like [Stack Overflow](https://stackoverflow.com/), where developers ask common questions and other developers provide detailed responses, either out of altruism, reputation-building, or simply for the satisfaction of helping their peers.

There's been speculation recently that ChatGPT is siphoning traffic from Stack Overflow. Some [sources](https://www.similarweb.com/blog/insights/ai-news/stack-overflow-chatgpt/) suggest this based on observed trends, but I personally find the evidence inconclusive. Regardless, anecdotally speaking, since the advent of ChatGPT, I've found less of a need to visit Stack Overflow. Why wait for human responses when ChatGPT can instantly help me solve problems or even write the code for me?

This shift in behavior could cause a "freeze" in the development of Stack Overflow. If developers stop asking questions, there will be fewer responses and less novel human knowledge being published. This scenario arises from a simple question of incentives - why wait for a response on Stack Overflow when you can get an instant answer from ChatGPT?

Now, it's crucial to acknowledge that if an AI model like ChatGPT cannot learn new things, that would indeed be a problem. We've already categorized knowledge into two types: static and dynamic. We can generally assume that LLMs like ChatGPT have sufficiently grasped static knowledge, but when it comes to dynamic knowledge, we need a mechanism for updating the model with new information.

Let's consider an example: [Mojo](https://www.modular.com/mojo), a new programming language built specifically for machine learning tasks. Mojo is a superset of Python, so the basic principles of data structures and algorithms remain the same, but the syntax, semantics, and performance features are new. How can we enable ChatGPT to read and debug Mojo?

For reading, the answer might be simple: include the Mojo [documentation](https://docs.modular.com/mojo/) in the model’s training set. This provides a basis for understanding the syntax and semantics of Mojo.

But for debugging, the solution could be more nuanced. With tools like [GitHub Copilot X](https://github.com/features/preview/copilot-x), which is powered by an LLM, your codebase and terminal context can be streamed into the model. As a result, while you're using Mojo, you are implicitly providing the LLM with valuable data about workflows, patterns, and issues inherent in programming with this new language. Notably, you are doing so while paying a subscription fee for the service!

Another scenario to consider is how an LLM might learn about dynamic events like a war. We don't yet have a system capable of capturing and processing real-time audio, video, or other sensor data into a form that can be used by LLMs. While a working prototype of such a system is likely feasible and inevitable over the next decade, we still rely on humans to translate complex inputs (e.g., on-the-ground reporting) into outputs (e.g., news articles) that can feed into LLM training.

**In summary:**
- LLMs can learn implicitly from captured data, as seen in the case of programming languages.
- For data we can't yet capture effectively, LLMs will need explicit inputs, as is the case with dynamic events like wars.
- As the technology for capturing inputs improves, we can progressively automate this process.
- In the meantime, humans will continue to focus on the areas where LLMs fall short, synthesizing complex inputs into outputs to create a rich and diverse (input, output) dataset.
- This dynamic relationship between human efforts and AI capabilities will continue to drive the evolution and improvement of large language models.

## What will we do with human-LLM conversations?

As of today, ChatGPT boasts a staggering [100 million users](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/), resulting in an extensive volume of human-LLM conversations. The question then arises: what do we do with these interactions? As with most things in AI, it comes back to data. What valuable data can be gleaned from these conversations, and how can OpenAI utilize this information? Let's examine a few possibilities.

**Product discovery**: One approach could involve summarizing the conversations, embedding the summaries, and subsequently clustering them. By identifying the centroid of each cluster and generating a descriptive label for the cluster using a generative model, we can start to understand what people are primarily using the model for. Furthermore, ChatGPT could be trained to proactively ask users for feedback or suggestions, gauging their interest in potential new products.

**Reinforcement learning**: Another approach might involve using conversation outcomes as learning signals for reinforcement learning. For instance, human feedback could label successful and unsuccessful conversations, and these labels could be used as rewards in reinforcement learning algorithms, such as Reinforcement Learning from Human Feedback (RLHF). This feedback could also be supplemented by binary labels generated by the model itself.

**Learning optimal prompts**: Lastly, by associating tasks with clusters of conversations and their descriptions, the system could learn to generate optimal prompts for those tasks. A hypothetical workflow might involve generating a prompt, having two LLMs engage in a conversation based on this prompt, scoring the result using our reward signal, and updating the prompt-generation policy using reinforcement learning methods like Proximal Policy Optimization (PPO), similar to what's used in the RLHF step of training ChatGPT.

**In summary:**
- The sheer volume of human-LLM interactions can be leveraged to:
    - Enhance the user experience by understanding and helping users accomplish their existing goals more effectively.
    - Improve the model itself, based on conversation outcomes and reward signals.
    - Discover and prioritize novel applications or products by analyzing usage patterns and directly soliciting user feedback.

## How do we solve the hallucination problem?

The challenge of hallucination — where generative models invent content rather than accurately generating information from their training data — is a critical hurdle to clear in the advancement of Language Learning Models (LLMs). One possible solution has occurred to me recently, influenced as I am by my current work in the field of cryptocurrency and my previous experience in machine learning. I don't present this as the definitive answer, but merely as an intriguing idea that draws on the principles of decentralized systems.

Retrieval models, which select fixed outputs from a fixed 'menu' of options, inherently navigate around the hallucination problem by explicitly restricting the model's potential responses. Generative models, on the other hand, create new content *ex nihilo*, and in doing so relinquish some control over what is produced — potentially leading to hallucinations.

At its core, the LLM problem is a data problem. If we could enlist human annotators to rectify hallucinations on a large scale, we could directly teach the model to avoid producing such outputs. However, an important question arises: ***who*** gets to decide what is an acceptable output? Currently, that's OpenAI — analogous to how Facebook was the arbiter of acceptable content during the 2020 presidential election.

This sparks an even deeper question: How do we resolve the hallucination problem ***without*** a centralized authority? That is, how do we develop models whose voice genuinely echoes the consensus of the larger community?

The tech world has been wrestling with a similar question for the last ~15 years, particularly in terms of monetary systems. The challenge is to create a trustworthy system for transferring monetary value that doesn't hinge on centralized, potentially biased, intermediaries.

These concurrent threads of thought led me to a conceptual crossroads. An interesting avenue to explore, in the spirit of thought experimentation, might look something like this:

The blockchain consensus algorithm known as Proof of Stake (PoS) has validators propose and verify blocks based on the number of tokens they're willing to stake. Proposing valid blocks earns rewards; proposing fraudulent ones incurs penalties, including losing their stake.

Adapting a PoS-like structure to LLMs might involve:
- Users as validators, offering feedback on the model's outputs.
- Validators having something at stake — be it points, reputation, privileges, or digital currency.
- Validators providing accurate feedback would be rewarded.
- Those giving inaccurate feedback or behaving maliciously would lose their stake.

Such a system could encourage accurate feedback and dissuade detrimental actions. Still, it's not without challenges. There's the risk of exploitation by coordinated groups and the requisite for validators to have adequate expertise to provide accurate validation. And, of course, it's vital to have a diverse and balanced group of validators to avoid biases.

**In summary:**

- This contemplation is just one of numerous potential solutions to the hallucination problem, inspired by the fusion of decentralized systems and machine learning principles.
- As the hallucination issue is fundamentally a data problem, any viable solutions need to consider how data can be accurately generated, scrutinized, and managed.
- Decentralized consensus systems might provide a fresh perspective on how to maintain the quality and integrity of generative models without relying on a centralized authority.

## How will we compensate data providers?

In the vast world of data, there exist two predominant categories of providers: **platforms or companies** that aggregate data (like Reddit, Stack Overflow, Mastercard, Facebook), and **individuals** who are the original creators of such data. As the landscape of data usage shifts, particularly with the advent of Language Learning Models (LLMs), so too will the compensation models for these providers.

**Current Compensation Paradigms:**

At present, data is mainly consumed for advertising, with companies acquiring it to better understand and target consumer behavior. Compensation for this data generally falls into three categories:

- **Financial**: Direct payments or royalties serve as the most straightforward form of compensation. Yet, complications arise due to copyright laws and the ease with which digital content can be replicated, diminishing the monetary value of the original content.

- **Reputation**: Platforms like Quora or Stack Overflow award individuals with recognition for quality contributions, thus enhancing their personal brand or standing within a community.

- **Moral**: Individuals may derive satisfaction from contributing unique content or insights to the world. This form of compensation grants a sense of purpose and validation.

**The Impact of LLMs on Compensation:**

As LLMs become more prevalent, demand for data is likely to shift from purely consumer-targeting information towards a more comprehensive understanding of the world — problem-solving techniques, programming languages, writing styles, current events, and more. This evolution will necessitate changes to existing compensation models:

- **For Companies**: In the LLM world, companies that provide data will likely continue to be [financially compensated](https://www.wired.com/story/stack-overflow-will-charge-ai-giants-for-training-data/), maintaining a familiar model.

- **For Individuals**: The LLM landscape will introduce new dynamics around individual compensation:

    - **Financial**: The same challenges persist, particularly the dilution of content's monetary value due to easy replication.

    - **Reputation**: The potential for reputational benefits increases. If LLMs can credit their sources, individuals who initially provided key information could garner more widespread recognition. Rather than being ***the*** source on Quora, they could be known as ***the*** source behind the LLM, which might be one of just a few do-it-all AI entities.

    - **Moral**: The sense of moral satisfaction might be undermined, as LLMs' capacity to replicate and generate human-like content could diminish the feeling of individual uniqueness and the joy of creation.

**In summary:**
In the long term, the rise of LLMs could fundamentally alter the dynamics of data provision and consumption. It presents a complex blend of opportunities and challenges, potentially devaluing individual contributions by making human creators superfluous for the ongoing generation of content. This shift prompts a deeper philosophical question about the significance of individual uniqueness and human contribution in a world of digital content. While the benefits of LLMs — such as increased efficiency and wider knowledge distribution — are appealing, the potential impacts on individual creators and their sense of value and contribution warrant careful contemplation.

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
```