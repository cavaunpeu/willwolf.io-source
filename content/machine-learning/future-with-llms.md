Title: Our Future with LLMs
Date: 2023-08-19 17:00
Author: Will Wolf
Lang: en
Slug: future-with-llms
Status: published
Summary: In this post, I explore the evolving world of Language Learning Models (LLMs), considering how they learn, the future of human-LLM conversations, the hallucination problem, compensating data providers, the potential lucrativeness of data annotation, and the advent of a new Marxist struggle.
Image: ../images/future-with-llms/human-mind.png

# What is a Language Model?

To introduce the language model let's talk about human speech. If I were to say, "The boy put on his boots and went to the...," what word comes next? Well, there are many that fit. As an English speaker, you could surely list a few: "store," "park," "bar," even "best"—as in "best ice cream spot in town." Conversely, many words could *never* follow this phrase, like "don't," "photosynthesis," or "trigonometry."

And how do you know this? How can you be sure? It's because you have a "model of language" in your brain, a "language model" that you've acquired over your lifetime. The more language you ingest through interacting with the world the better your model becomes.

# How does ChatGPT know so much?

In the past 30 years, internet users have unwittingly built the largest, broadest, most diverse, most interesting dataset in human history from which to learn machine learning models. These data come in various forms, or "modalities," like images from Instagram, videos from YouTube, audio from various platforms, and text from Wikipedia, Reddit, and more.

ChatGPT is learned from text. Whereas you've trained your language model only on the language you've encountered in the handful of decades for which you've been alive, ChatGPT has been trained on a large chunk of all text ever written on the internet, period. For comparison, Quora[^27] users estimate that this would take a human roughly "23.8 million years" to "you can't, so it's an impossible question" to "you would be dead long before you even made a start." This makes it very good at predicting the next word in a phrase, such as our earlier example about the boy and his boots. More generally, it can skillfully continue almost *any* text, such as "Can you recite me the Indonesian national anthem in the style of Shakespeare?" or "What should I make for dinner if I only have salmon and chocolate?" or "What's the best way to get to the moon?"

# How does ChatGPT work?

Creating ChatGPT involves three steps[^20]:

1. **Train a language model:** Given a phrase, teach the model how to output, i.e. "predict," the next word. (Repeating this process ad infinitum, and appending the predicted word to the end of the initial phrase each time, it can generate a complete response.)
2. **Fine-tune on `(prompt, response)` pairs:** Humans provide both parts of these pairs, giving concrete demonstrations of how to complete the tasks the model will be asked to perform.
3. **Further fine-tune via a model of output quality.** Humans rate the quality of ChatGPT's outputs, then a second model learns these relationships, then ChatGPT learns to output high-quality responses via this second model. This process is known as "Reinforcement Learning from Human Feedback"[^24] (RLHF).

# This is a story about data

Throughout my career, I've learned almost every ML problem is a story about data. Where does it come from? What's wrong with it? How much does it cost? How do we de-bias it? How do we get more of it? Which data should we label next? And on. ChatGPT is no different.

With this in mind, here are a few keys points regarding where we stand today:

**Static knowledge**: ChatGPT's language model (GPT-4) has been trained on a large chunk of the written internet, dated through September 2021 (as users will now know by heart, as the system loves to restate this limitation). Encapsulated in these data is the knowledge required to solve a substantial number of *static-knowledge* tasks. For example, the model can summarize news articles; as the nature of summarization doesn't really evolve year over year, the model does not per se require additional data to accomplish this task. It has enough already.

**Dynamic knowledge**: Conversely, imagine that we'd like to translate classical Greek into modern English. Well, what does "modern" mean? Language constantly evolves[^28] to include new vocabularly and modes of expression. So, while the fundamentals of translation don't really change, the contemporary details do. To keep pace with these details, the model needs to be updated with examples of this text. Ten years ago, I surely wasn't saying "that's lit" myself.

**Novel knowledge**: Finally, novel knowledge defines the set of tasks or abilities that the model has never encountered. For instance, a novel discovery in physics, e.g. room-temperature superconductivity[^29] is an example of *dynamic knowledge* if this work is an *extension* of the scientific knowledge, logical reasoning, historical expectations, etc. that the model already posseses. Conversely, this discovery is an example of *novel knowledge* if it is predominantly composed of never-before-seen ways of conceptualizing the world, e.g. "a new mathematics," alien anatomy, etc.

The vast majority of knowledge is either static or dynamic. However, for completeness, we leave a small sliver of space for novel knowledge as well.

**Human annotators**: Human annotators (paid and trained by OpenAI) have provided the data required for the supervised fine-tuning and RLHF steps. Should we wish to expand the "foundational" set of tasks that we explicitly want the model to solve, or update our "preference" regarding the way in which the model expresses itself, we'll need more annotations.

# A menu of questions

In this post, I explore our future with LLMs from the perspective of *data*. I'll do so by asking and answering a series of questions—touching on the methods, the players, the economics, and the power struggles that potentially lie ahead.

1. **How will LLMs learn new information?**
2. **What will we do with human-LLM conversations?**
3. **How do we solve the "hallucination" problem?**
4. **How will we compensate data providers?**
5. **Will data annotation be lucrative?**
6. **Is this the new Marxist struggle?**

Let's begin.

## How will LLMs learn new information?

I work as a software engineer. If you don't, you might believe that all engineers have committed to memory the knowledge and syntax required to solve every task we ever encounter. Alas, we haven't. Instead, we commonly use "question-answer" sites like Stack Overflow[^30] to see how other developers have solved the problem at hand. Before ChatGPT, I used Stack Overflow almost daily.

Several sources posit[^31] that ChatGPT is directly cannibalizing Stack Overflow traffic. Personally, I don't find the statistics they provide particularly convincing. So, let's use an anecdote instead: since I started using ChatGPT in ~4 months ago, I have not been on Stack Overflow once. Why wait for human responses when ChatGPT responds instantly?

If other developers do the same, Stack Overflow "freezes." In other words, no new human programming knowledge is published at all. Like language translation, coding implies translating a set of logical expressions in one's brain into machine-readable instructions. Across the spectrum of present and future programming languages, the fundamentals generally don't change. However, the details do. In this vein, being a software engineer (as a human, or an AI model) is a dynamic-knowledge task. When the next programming language hits the market, how will developers know how to use it?

Let's consider Mojo[^32], a new programming language built specifically for AI developers. As Mojo is a superset of Python, our knowledge about Python will still apply. However, Mojo will bring new features that we haven't seen before. Simply put, how will ChatGPT learn how to read and debug Mojo?

For reading, the answer might be simple: include the Mojo documentation[^33] in the model's training set. This provides a basis for understanding the syntax and semantics of Mojo.

For debugging, it's more subtle. GitHub Copilot X[^34]—the LLM-powered tool that helps you write and debug code—will now capture and send your terminal context back to OpenAI itself. As such, with this beta, the LLM is actively "acquiring new data" on the workflows, questions, patterns, etc. inherent in programming in Mojo.

Taken together, for the model to update its understanding of dynamic and novel knowledge tasks, we must provide it with new data. In what follows, we codify the nature of this provision along three main axes: implicit vs. explicit, quality, and velocity.

### Implicit vs. explicit

The nature of data provision will range from implicit to explicit. Capturing the Mojo developer's terminal context is an example of implicit data provision. Curating model training examples about how to resolve common Mojo errors is an example of explicit data provision. Answering empirical questions on Stack Overflow has elements of both.

Broadly, implicit data will be easier to collect as there's more of it to go around, and vice versa.

### Quality

In the three cases just mentioned, we suppose that humans act "rationally," meaning that they earnestly try to produce the "right information" in order to solve their problem. However, the quality of these data varies across each case—as a function of who is providing it, and what their incentives and requirements are.

In the "capturing terminal context" setting—implemented naively—we are capturing information from *all* developers. Some might be good, others bad. While most developers are likely "trying to solve their problem," or "debugging code until it works," the quality of this information will vary as a function of their skills.

In Stack Overflow, the same "available to all" feature applies. However, there is both an additional social pressure placed on users of the site to provide correct information—people don't want to look silly in front of their peers—as well as an explicit feedback mechanism—answers deemed correct get "upvoted," and vice versa. Nominally, these constraints increase data quality.

Finally, we assume that the "manually curate training set examples" setting gives the highest quality data of the three. Why? A company is paying a human to explicitly teach the model information. Before annotation, they ensure this human has the right qualifications; after annotation, they likely review the results. Taken together, the more training and scrutiny, the higher the quality.

### Velocity

Finally, across the three settings, the *speed* with which we can generate a large and diverse dataset varies. In manual curation, it's slowest (single human). On Stack Overflow (many human-human interactions), it's faster. In the terminal (many human-machine interactions), it's fastest (and probably by a lot). A "many machine-machine interactions" setup, e.g. copies of a reinforcement learning system where an AI plays the part of the programmer, gets feedback from the language, and iterates, all running in parallel, would be even faster...

### So where do I get data?

With the above in mind, companies will seek out data sources that make an optimal trade-off between: the pragmatic implications of collecting implicit vs. explicit data, the quality of the data provided, and the speed with which it is generated. Broadly, implicit data will be easier to collect, of lower quality, and higher velocity. Explicit data will be harder to collect, of higher quality, and lower velocity.

**Overall, companies will need to identify the "data provision venue" that makes the right trade-offs for them and their model.** Then, they'll need to be strategic about how to "drop their net" into this stream and catch the passing fish.

## What will we do with human-LLM conversations?

ChatGPT already boasts a staggering 100 million users[^35]. An enormous quantity of human-LLM conversations are taking place daily. What do we do with these interactions? What valuable data can be gleaned from these conversations, and how can OpenAI use this information?

### Product discovery

To start, OpenAI could use these conversations to determine how people are using the model, and what new products to therefore build. For instance, we could summarize the conversations, embed the summaries, cluster them, then use the cluster's centroid to generate a cluster label. In this way, we can start to understand what people are using the model for. Furthermore, ChatGPT could be trained to proactively ask users for feedback or suggestions, gauging their interest in potential new features or products.

### Reinforcement learning

In another approach, we could use conversation outcomes as signals for reinforcement learning. For instance, annotators could label successful and unsuccessful conversations, and these labels could be used as rewards in algorithms like RLHF. This feedback could also be supplemented by binary labels generated by the model itself.

### Learning optimal prompts

Lastly, by associating tasks with clusters of conversations and their descriptions, the system could learn to generate optimal prompts for those tasks. For instance, we could generate a prompt, have two LLMs engage in a conversation based on this prompt, score the result using our reward model, then update the prompt-generation policy itself.

**Taken together, companies will use human-LLM conversations to** among other things, discover and prioritize novel applications and products, improve the model, and improve the experience of using the model itself.

## How do we solve the hallucination problem?

"Hallucination" is when an LLM says things that have no basis in fact or reality. If we knew *when* the model did this, we could simply restrict those outputs; if we knew *why*, we could design better models that hallucinate less. Unfortunately, the answers to these questions remain elusive[^17].

Retrieval[^37] models select outputs from a fixed "menu" of choices. In this way, we implicitly "solve" the hallucination problem by explicitly restricting a priori what the model can and can't "say." Generative models, on the other hand, make a different trade-off: by allowing the model to generate novel content *ex nihilo*, we forfeit some of this control.

Paying rational human annotators to "correct" all recorded hallucinations would likely improve this situation. However, as history has shown, policing the actions and behaviors of every constituent is not a scalable strategy. In addition, the question of *who* decides what "correct" actually means remains open for debate. In the context of ChatGPT, this is OpenAI. Similarly, in the context of the 2020 presidential election, it was Facebook that decided what content was and was not acceptable to promote. Combining the two, an interesting question arises: How do we solve the hallucination problem without a centralized authority? Said differently, how do we build models whose voice represents that of the broader consensus? It is extremely likely that a (heated) discussion surrounding some form of this question will unfold in the coming years.

My technical background is largely in machine learning. However, I've been working in crypto for the past two years. In this section, I'll borrow an idea from the latter and apply it to the former. The following idea may be fanciful and impractical and is not the only way to approach this problem. Nonetheless, it makes for an interesting thought experiment.

### Proof of Stake

The crypto world has spent the last ~15 years trying to answer a similar question: How do we build a scalable, trustworthy system for the digital transfer of monetary value that does not rely on centralized intermediaries? To date, one of the key mechanisms used to achieve this end is Proof of Stake (PoS). PoS is a consensus algorithm where participants, or "validators," are collectively entrusted to verify the legitimacy of transactions. To incentivize earnest behavior, PoS employs the following mechanism:

- Participants are paid to validate transactions.
- Prior to validating transactions, participants "stake" capital. This "stake" is like a "hold" placed on your credit card when renting a car.
- The more capital you "stake," the more likely you are to be selected to validate transactions.
- If other participants deem your behavior dishonest, your "stake" is taken (and you do not get paid).

Taken together, PoS promotes transaction fidelity with economic incentives and penalties. Dishonest participants may lose their staked tokens, creating a self-regulating system where everyone has a vested interest in success.

### Applying Proof of Stake to LLMs

How might we apply PoS to LLMs? In effect, the users are the validators who ensure the legitimacy not of transactions, but of model outputs. Validators would have a stake—such as points, reputation, privileges, or digital currency—at risk, creating a vested interest in the accuracy of feedback. Then, model outputs would be periodically selected for collective review. Validators would be rewarded for proposing valid feedback in consensus with other users. Conversely, those providing inaccurate feedback or acting maliciously would lose their stake.

Much like PoS in the blockchain world, this system is not without its challenges. For instance:

- How do we ensure that a small number of "high-stake" users don't control the system?
- Will the collective expertise of users empirically "satisfy" the model provider? If it's Elon, maybe; if it's Sam Altman, unclear.
- What balance between rewards and penalties promotes truthful feedback yet does not stifle participation?
- Etc.

Ultimately, whether or when this type of system might be introduced hinges on the question of who retains power. Will LLMs simply be services offered by private companies? Will governments mandate their use as a public utility informed by, and built for, the body politic itself? I don't have the answers to any of these questions. However, my popcorn is ready.

**Overall, Proof of Stake is but one approach to solving the hallucination problem.** As an algorithm for decentralized consensus, its relevance will evolve with ongoing narratives surrounding scalability, fairness, and the distribution of power in the context of LLMs.

## How will we compensate data providers?

Much like LLMs, business is a story about data as well. For instance, historical customer purchase data allows a business to prioritize which products and services to sell. Customer attributes like age, location, gender, and political preference enable more targeted advertisements. Data collected in feedback forms hint at new products that customers might buy.

Before the internet, companies employed a variety of traditional methods to collect these data: questionnaires or comment cards, tracking purchases and preferences through loyalty cards, and utilizing demographic information and other publicly available data. With the internet, the game changed overnight. Now, companies can track every page visit, click, and even eye movements[^37]. In addition, they encourage the creation of data expressing *implicit* user preferences, such as pictures on Instagram, chats between friends on Messenger, videos on YouTube. Much like before, these data offer clear value to businesses.

Other types of data are valuable as well. Expert replies on question-answer sites like Quora offer informational value to users, and garner reputational value for the author herself. Basic data annotation enables machine learning practitioners to train models. In this[^38] episode of The Daily, Sheera Frenkel discusses how fantasy story writing communities give a sense of purpose and satisfaction to their writers (and entertainment value to their readers). Finally, online courses on sites like [Platzi](https://platzi.com/) offer clear educational value to students.

Overall, these data remain valuable to diverse parties in myriad ways.

### Current compensation models

In exchange for the data above, its creators are compensated through the following three means:

- **Financial**: Direct payments, royalties, or subscription fees for music, art, literature, educational material, data annotation, and more.

- **Reputation**: On platforms like Quora or Stack Overflow, individuals gain recognition based on the quality of their responses, knowledge, and expertise, enhancing their personal brand within a community.

- **Spiritual**: Individuals derive personal satisfaction from contributing something unique to the world.

### What do LLMs providers value?

LLMs represent an entirely new consumer of data. As such, given the ways these models will be used, LLM providers value different things. Already, "beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance."[^18] Conversely, beyond hallucination, what are these systems *not* good at? What do they need to be maximally useful and achieve widespread user adoption? Finally, how will model providers compensate those that provide the requisite data?

**New information**: First and foremost, LLMs need to learn new information as detailed above. In exchange for this information, model providers could compensate data creators financially. For instance, OpenAI might (continue[^39] to) pay Reddit for its forum data or pay the NYT for the articles its staff writes.

**Multi-turn feedback**: Responding to factoid questions is easy; maintaining a coherent and intuitive conversation over multiple "turns" is more difficult. As LLMs are increasingly used for conversational use cases, this type of data becomes more relevant. Ironically, the predominant place where these data will likely be created are in human-LLM conversations themselves. As such, model providers may offer free usage of their services in exchange for these data, neatly mirroring the "free but invasive but no one cares" playbook that Facebook and Google have perfected.

**Answering subjective questions**: When answering a subjective question, a model should provide one or more credible viewpoints in response. For instance, "Why did the United States experience inflation in 2021 and onward?" should be addressed with the diverse perspectives of capable economists. Irrespective of who these economists are chosen to be, it's clear that the "airtime" they will receive will be immense. As such, being *the* person featured by *the* LLM offers significant reputational benefits, much like being the top search result on Google in years past.

### Future compensation models

Taken together, future compensation models might look as follows:

- **Financial**: Direct payment for data dumps, application usage data, etc. In addition, data providers may achieve (micro) royalty[^19] payments[^25] every time their data are referenced in a model output.

- **Reputation**: Becoming *the* response from *the* LLM offers similar benefit to being a "top answer" on today's question-answer sites. To wit, GitHub Copilot is already[^40] implementing such "attribution" features.

- **Spiritual**: This form of compensation may really change. As we "share our unique voice" with the LLM, e.g. a short story that we've written, the model can effectively "impersonate" us thereafter—forever. Will this "digital immortality" inspire feelings of personal grandeur? Or despair for the fact that we're "no longer needed?" Similarly, how will people feel interacting with an intelligence superior to their own? These questions are highly uncertain and will find answers in time.

**Overall, models for data compensation in a world with LLMs will remain similar to those of today.** However, the spiritual experience of contributing data to and interacting with these models will evolve materially, challenging our perceptions of identity, purpose, and place in the years to come.

## Will data annotation be lucrative?

Machine learning models are trained to accomplish specific tasks. To do this, humans first "label some data," i.e. perform this task up front. Next, we train the model to learn relationships between its inputs and these labels. For instance, when training a model to distinguish between "cat" and "dog" images, it learns from a dataset where each photo had been explicitly labeled by a human as containing either a "cat" or "dog." Similarly, an autonomous vehicle capable of resolving sophisticated moral dilemmas implies that humans imparted this judgement during training. As we ambition to build models that accomplish increasingly sophisticated tasks, data annotation complexity will increase in turn.

As noted above, ChatGPT relies on two types of data annotation: diverse task demonstrations, e.g. given a prompt, "act out" both sides of the human-machine conversation, as well as model output quality ranking. In effect, these annotations control how the model should respond as well as which types of responses are preferred. In other words, they control a lot! As such, to use this model effectively in increasingly "high stakes" corners of society, e.g. as behavioral therapists, legal counsel, tax attorneys, etc., we may require annotations of increasingly nuanced ethical, moral, "human" substance. Taken to its extreme, we can imagine these annotators as philosophers, legal scholars, and religious authorities, i.e. highly-trained specialists commanding due compensation.

With this in mind, numerous questions arise: Who will appoint these annotators? Will private companies continue to train LLMs exclusively? Will governments mandate a certain participation model, e.g. "democratic" annotation, with each citizen or interest group having a say? Financial compensation aside, will annotators gain status a la Supreme Court justices?

**Overall, highly-specialized data annotation will be in demand.** Time will tell if this translates into a financially lucrative career path.

## Is this the new Marxist struggle?

Viewed through the lens of Marxism, model providers like OpenAI may well ascend to the role of the bourgeoisie of our time—the esteemed custodians of the means of production. Conversely, data providers represent the proletariat; without their digital labor, the "factory lights" would dim. Will their relationship become the next flashpoint of class struggle in our digital age?

To begin, let's examine the relationship between social media platforms and their users over the past ~20 years. In effect: the former amassed vast fortunes through the unpaid efforts of the latter: users supplied data, which the platforms then leveraged for targeted advertising. In return, users received free access to services that allowed them to connect with friends, organize their professional lives, and access information online. While this arrangement sparked outrage among some, dissent eventually faded.

The advent of LLMs, however, introduces a crucial difference: the dramatically expanding range of AI applications. The services promised by LLMs aren't confined to social communication or productivity tools. Instead, they're likely to fill roles as diverse as accountants, therapists, fitness instructors, educators, career coaches, among many others. As AI permeates deeper into our daily lives, the issues of labor and compensation in data provision become that much more salient.

The outcome of this tension will be heavily informed by the answer to the following question: will AI make "all boats rise," or will it swallow them whole? In the former scenario, people lose jobs, retrain, and are re-employed in the next generation of professional careers. In the latter, we devolve into the dystopia depicted in Yuval Noah Harari's "Homo Deus," in which a sizeable portion of the labor force has nothing of "economic relevance" to contribute at all.

What kind of jobs that can be done by LLMs? In fact, the answer to this question is synonymous with the data on which the model was trained. As such, "knowledge worker" jobs based in text (writing, coding, etc.) stand threatened. In addition—and while this post has covered text-only LLMs alone—these models will invariably trend towards understanding *multiple* digital modalities simultaneously—accepting inputs and producing outputs in the form of text, image, video, audio, and more. These systems will work because of the volume of such data on which we can train, putting the creators of that data—software engineers, digital artists, YouTube content creators, etc.—at risk.

Crucially, though, this type of work is one among many. To date, jobs in fields such as healthcare, hospitality, craftsmanship, maintenance and repair, agriculture, emergency services, beauty, and education are not at real risk of replacement by AI. As such, we may simply see displaced knowledge workers "diffuse" into these other spheres, postponing real conflict. From there, the cycle might repeat: humans create data, model providers collect that data, then train models, then replace humans. And finally, once there are no "untouched" sectors of the economy left, we'll be able to more clearly perceive an answer to our question.

**Taken together, I don't envision a material class conflict anytime soon.** AI has a lot further to go towards Harari's dystopia for this to happen. For the time being, the relationship between model providers and data producers will remain largely peaceful, and endlessly interesting.

# Conclusion

The LLM story is sure to evolve quickly. It's unclear where it will go. In the best case, human knowledge, satisfaction, and general well-being compound exponentially. My fingers are crossed.

# References

[^1]: @misc{2210.11610,
Author={Jiaxin Huang and Shixiang Shane Gu and Le Hou and Yuexin Wu and Xuezhi Wang and Hongkun Yu and Jiawei Han},
Title={Large Language Models Can Self-Improve},
Year={2022},
Eprint={arXiv:2210.11610},
}

[^2]: @article{dave2023stackoverflow,
  title={Stack Overflow Will Charge AI Giants for Training Data},
  author={Dave, Paresh},
  journal={Wired},
  year={2023},
  month={Apr},
  day={20},
  note={The programmer Q&A site joins Reddit in demanding compensation when its data is used to train algorithms and ChatGPT-style bots},
  url={https://www.wired.com/story/stack-overflow-will-charge-ai-giants-for-training-data/},
  timestamp={5:19 PM}
}

[^3]: @article{anirudh2023stackoverflow,
  title={Is This the Beginning of the End of Stack Overflow?},
  author={Anirudh, VK},
  journal={Analytics India Magazine},
  year={2023},
  month={Apr},
  day={18},
  note={Integrating an LLM into Stack Overflow won't make its problems disappear},
  url={https://analyticsindiamag.com/is-this-the-beginning-of-the-end-of-stack-overflow/},
  publisher={Endless Origins}
}

[^4]: @online{chandrasekar2023community,
  title={Community is the future of AI},
  author={Chandrasekar, Prashanth},
  year={2023},
  month={Apr},
  day={17},
  note={To keep knowledge open and accessible to all, we must come together to build the future of AI.},
  url={https://stackoverflow.blog/2023/04/17/community-is-the-future-of-ai/},
  publisher={Stack Overflow Blog},
  organization={Stack Overflow}
}

[^5]: @misc{2306.15774,
Author={Xiang 'Anthony' Chen and Jeff Burke and Ruofei Du and Matthew K. Hong and Jennifer Jacobs and Philippe Laban and Dingzeyu Li and Nanyun Peng and Karl D. D. Willis and Chien-Sheng Wu and Bolei Zhou},
Title={Next Steps for Human-Centered Generative AI: A Technical Perspective},
Year={2023},
Eprint={arXiv:2306.15774},
}

[^6]: @misc{2306.08302,
Author={Shirui Pan and Linhao Luo and Yufei Wang and Chen Chen and Jiapu Wang and Xindong Wu},
Title={Unifying Large Language Models and Knowledge Graphs: A Roadmap},
Year={2023},
Eprint={arXiv:2306.08302},
}

[^7]: @article{2305.18339,
Author={Yuntao Wang and Yanghe Pan and Miao Yan and Zhou Su and Tom H. Luan},
Title={A Survey on ChatGPT: AI-Generated Contents, Challenges, and Solutions},
Year={2023},
Eprint={arXiv:2305.18339},
Doi={10.1109/OJCS.2023.3300321},
}

[^8]: @misc{2108.13487,
Author={Shuohang Wang and Yang Liu and Yichong Xu and Chenguang Zhu and Michael Zeng},
Title={Want To Reduce Labeling Cost? GPT-3 Can Help},
Year={2021},
Eprint={arXiv:2108.13487},
}

[^9]: @misc{2307.10169,
Author={Jean Kaddour and Joshua Harris and Maximilian Mozes and Herbie Bradley and Roberta Raileanu and Robert McHardy},
Title={Challenges and Applications of Large Language Models},
Year={2023},
Eprint={arXiv:2307.10169},
}

[^10]: @misc{2212.10450,
Author={Bosheng Ding and Chengwei Qin and Linlin Liu and Yew Ken Chia and Shafiq Joty and Boyang Li and Lidong Bing},
Title={Is GPT-3 a Good Data Annotator?},
Year={2022},
Eprint={arXiv:2212.10450},
}

[^11]: @misc{2306.11644,
Author={Suriya Gunasekar and Yi Zhang and Jyoti Aneja and Caio César Teodoro Mendes and Allie Del Giorno and Sivakanth Gopi and Mojan Javaheripi and Piero Kauffmann and Gustavo de Rosa and Olli Saarikivi and Adil Salim and Shital Shah and Harkirat Singh Behl and Xin Wang and Sébastien Bubeck and Ronen Eldan and Adam Tauman Kalai and Yin Tat Lee and Yuanzhi Li},
Title={Textbooks Are All You Need},
Year={2023},
Eprint={arXiv:2306.11644},
}

[^12]: @inproceedings{yoo2021gpt3mix,
  title={GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation},
  author={Yoo, Kang Min and Park, Dongju and Kang, Jaewook and Lee, Sang-Woo and Park, Woomyeong},
  pages={192},
  year={2021},
  organization={NAVER AI Lab and NAVER Clova AI},
  address={NAVER AI Lab and NAVER Clova AI},
  email={{kangmin.yoo, dongju.park, jaewook.kang}@navercorp.com, {sang.woo.lee, max.park}@navercorp.com},
  url={https://aclanthology.org/2021.findings-emnlp.192.pdf}
}

[^13]: @misc{2302.13007,
Author={Haixing Dai and Zhengliang Liu and Wenxiong Liao and Xiaoke Huang and Yihan Cao and Zihao Wu and Lin Zhao and Shaochen Xu and Wei Liu and Ninghao Liu and Sheng Li and Dajiang Zhu and Hongmin Cai and Lichao Sun and Quanzheng Li and Dinggang Shen and Tianming Liu and Xiang Li},
Title={AugGPT: Leveraging ChatGPT for Text Data Augmentation},
Year={2023},
Eprint={arXiv:2302.13007},
}

[^14]: @misc{1707.06347,
Author={John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
Title={Proximal Policy Optimization Algorithms},
Year={2017},
Eprint={arXiv:1707.06347},
}

[^15]: @misc{2304.01852,
Author={Yiheng Liu and Tianle Han and Siyuan Ma and Jiayue Zhang and Yuanyuan Yang and Jiaming Tian and Hao He and Antong Li and Mengshen He and Zhengliang Liu and Zihao Wu and Dajiang Zhu and Xiang Li and Ning Qiang and Dingang Shen and Tianming Liu and Bao Ge},
Title={Summary of ChatGPT/GPT-4 Research and Perspective Towards the Future of Large Language Models},
Year={2023},
Eprint={arXiv:2304.01852},
}

[^16]: @misc{2209.01538,
Author={Xin Mu and Ming Pang and Feida Zhu},
Title={Data Provenance via Differential Auditing},
Year={2022},
Eprint={arXiv:2209.01538},
}

[^17]: @misc{2304.00612,
Author={Samuel R. Bowman},
Title={Eight Things to Know about Large Language Models},
Year={2023},
Eprint={arXiv:2304.00612},
}

[^18]: @misc{2303.12712,
Author={Sébastien Bubeck and Varun Chandrasekaran and Ronen Eldan and Johannes Gehrke and Eric Horvitz and Ece Kamar and Peter Lee and Yin Tat Lee and Yuanzhi Li and Scott Lundberg and Harsha Nori and Hamid Palangi and Marco Tulio Ribeiro and Yi Zhang},
Title={Sparks of Artificial General Intelligence: Early experiments with GPT-4},
Year={2023},
Eprint={arXiv:2303.12712},
}

[^19]: @book{lanier2013future,
  title={Who owns the future?},
  author={Lanier, Jaron},
  year={2013},
  publisher={Simon & Schuster}
}

[^20]: @online{openai2022chatgpt,
  title={Introducing ChatGPT},
  author={OpenAI},
  year={2022},
  month={11},
  day={30},
  url={https://openai.com/blog/chatgpt}
}

[^21]: @online{openai2022instruction,
  title={Aligning language models to follow instructions},
  author={OpenAI},
  year={2022},
  month={01},
  day={27},
  url={https://openai.com/research/instruction-following}
}

[^22]: @online{openai2017humanprefs,
  title={Learning from human preferences},
  author={OpenAI},
  year={2017},
  month={06},
  day={13},
  url={https://openai.com/research/learning-from-human-preferences}
}

[^23]: @online{openai2017ppo,
  title={Proximal Policy Optimization},
  author={OpenAI},
  year={2017},
  month={07},
  day={20},
  url={https://openai.com/research/openai-baselines-ppo}
}

[^24]: @online{huggingface2022rlhf,
  title={Illustrating Reinforcement Learning from Human Feedback (RLHF)},
  author={Nathan Lambert and Louis Castricato and Leandro von Werra and Alex Havrilla},
  year={2022},
  month={12},
  day={9},
  url={https://huggingface.co/blog/rlhf},
  organization={Hugging Face}
}

[^25]: @online{durmonski2023owns,
  title={Who Owns The Future? by Jaron Lanier [Actionable Summary]},
  author={Ivaylo Durmonski},
  year={2023},
  month={7},
  day={7},
  url={https://durmonski.com/book-summaries/who-owns-the-future/#5-lesson-2-ordinary-people-are-not-compensated-for-the-information-taken-from-them},
  organization={Durmonski.com},
  note={Actionable Book Summaries, Science & Tech Book Summaries}
}

[^26]: @article{bubeck2023sparks,
  title={Sparks of Artificial General Intelligence: Early experiments with GPT-4},
  author={Bubeck, Sébastien and Chandrasekaran, Varun and Eldan, Ronen and Gehrke, Johannes and Horvitz, Eric and Kamar, Ece and Lee, Peter and Lee, Yin Tat and Li, Yuanzhi and Lundberg, Scott and Nori, Harsha and Palangi, Hamid and Ribeiro, Marco Tulio and Zhang, Yi},
  year={2023},
  month={3},
  publisher={Microsoft Research},
  url={https://www.microsoft.com/en-us/research/publication/sparks-of-artificial-general-intelligence-early-experiments-with-gpt-4/}
}

[^27]: @misc{author2023howlong,
  title={How long would it take you to read the entire internet?},
  author={Stinson, Mark and Chovanek, Chris and Gibson, Jack},
  year={2023},
  howpublished={\url{https://www.quora.com/How-long-would-it-take-you-to-read-the-entire-internet}},
  note={Accessed: [insert date you accessed the link]}
}

[^28]: @book{mcculloch2019because,
  title={Because Internet: Understanding the New Rules of Language},
  author={McCulloch, Gretchen},
  year={2019},
  publisher={Hardcover},
  note={Published on July 23, 2019}
}

[^29]: @article{yirka2023korean,
  title={Korean team claims to have created the first room-temperature, ambient-pressure superconductor},
  author={Yirka, Bob},
  year={2023},
  month={July},
  day={27},
  journal={Phys.org},
  url={https://phys.org/news/2023-07-korean-team-room-temperature-ambient-pressure-superconductor.html}
}

[^30]: @misc{stackoverflow,
  title={Stack Overflow},
  year={2008},
  url={https://stackoverflow.com/}
}

[^31]: @online{carr2023stackoverflow,
  author={David F. Carr},
  title={Stack Overflow is ChatGPT Casualty: Traffic Down 14% in March},
  year={2023},
  url={https://www.similarweb.com/blog/insights/ai-news/stack-overflow-chatgpt/}
  month={April 19},
  update={Updated June 21, 2023}
}

[^32]: @online{modular2023mojo,
  title={Mojo 🔥 — a new programming language for all AI developers},
  year={2023},
  url={https://www.modular.com/mojo},
  publisher={Modular},
}

[^33]: @online{modulardocs2023mojo,
  title={Mojo Documentation},
  year={2023},
  url={https://docs.modular.com/mojo/},
  publisher={Modular},
}

[^34]: @online{github2023copilotx,
  title={Your AI pair programmer is leveling up},
  year={2023},
  url={https://github.com/features/preview/copilot-x},
  publisher={GitHub},
}

[^35]: @article{hu2023chatgpt,
  title={ChatGPT sets record for fastest-growing user base - analyst note},
  author={Hu, Krystal},
  journal={Reuters},
  year={2023},
  month={Feb},
  day={2},
  url={https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/}
}

[^36]: @misc{wikipedia2023information,
  title={Information retrieval},
  author={Wikipedia, The Free Encyclopedia},
  year={2023},
  note={Available: \url{https://en.wikipedia.org/wiki/Information_retrieval}}
}

[^37]: @online{lecomte2022,
  author={Patrick Lecomte},
  title={Companies are increasingly tracking eye movements — but is it ethical?},
  year={2022},
  url={https://theconversation.com/companies-are-increasingly-tracking-eye-movements-but-is-it-ethical-191842},
  note={Published: October 16, 2022 8.28am EDT},
  organization={The Conversation},
  institution={Université du Québec à Montréal (UQAM)},
  keywords={eye tracking, ethics}
}

[^38]: @misc{frenkel2023,
  author={Sheera Frenkel},
  title={The Writer's Revolt Against AI Companies},
  year={2023},
  note={Episode of The Daily},
  howpublished={Available at: \url{https://open.spotify.com/episode/26xt8MwmfaBlmU6GFjYumu}},
  organization={The New York Times},
  abstract={To refine their popular technology, new artificial intelligence platforms like Chat-GPT are gobbling up the work of authors, poets, comedians and actors — without their consent. Sheera Frenkel, a technology correspondent for The Times, explains why a rebellion is brewing.},
  keywords={AI, Chat-GPT, rebellion, technology, ethics}
}

[^39]: @online{ngila2023reddit,
  title={AI bots trained on Reddit conversations. Now Reddit wants to be paid for it.},
  author={Ngila, Faustine},
  year={2023},
  month={April},
  day={19},
  url={https://qz.com/reddit-ai-bots-training-payment-1850352526},
  publisher={Quartz}
}

[^40]: @online{salva2023introducing,
    author={Ryan J. Salva},
    title={Introducing code referencing for GitHub Copilot},
    year={2023},
    url={https://github.blog/2023-08-03-introducing-code-referencing-for-github-copilot/},
    month={August}
}