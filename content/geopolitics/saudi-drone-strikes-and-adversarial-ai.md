Title: On Saudi Drone Strikes and Adversarial AI
Date: 2019-09-22 08:30
Author: Will Wolf
Lang: en
Slug: saudi-drone-strikes-and-adversarial-ai
Alias: /geopolitics/2019/09/22/saudi-drone-strikes-and-adversarial-ai/
Status: published
Summary: In world of weaponized drones piloted by algorithms, what new strategic opportunities arise?
Image: ../images/saudi-adversarial/thumbnail.jpg

Last week, Houthi rebels—a militant political group in Yemen—claimed credit for exploding an oil refinery in Saudi Arabia. The explosion was the work of a missile, fired from a drone.

Though the group responsible is closely allied with Iran—and to wit, the American government does declaim Iran as responsible for facilitating the strike itself—it is by all accounts a non-state actor, notable for the following reasons:

1. As the cost of full-scale war increases due to, at a trivial minimum, the strength of weapons available, small, “surgical” operations become more common. This attack falls in directly with this trend.
2. You do not have to be the state to desire to affect world politics. As weapons become more potent, and the “density” of potential targets—population density, in the case of traditional military offensives, or density of nodes in an information-sharing network, in the case of cyber offensives, as two examples—increases, the ability to affect outsized change increases as well. As such, incentives for, and frequency of, strategic action from non-state actors will likely continue to rise—this being yet another example.
3. The technologies required to effectuate the attack in question are increasingly easy to obtain.

In sum, last week’s strike is yet another example of an isolated punch by a non-state actor with technologies (drones, at a minimum) ultimately available in the public domain. Moving forward, it seems safe to expect more of the same.

## Enter machine learning

Throughout history, much of war has been about destroying the enemy’s stuff. There are many ways to do this—each involving a distinct set of tradeoffs.

To this effect, drones are particularly interesting: they allow for precise, close-proximity strikes, without risking the lives of the aggressors. Presently, the majority of such drones are piloted remotely by humans; moving forward, they—in conceptual step with, say, self-driving cars—will pilot themselves, allowing for larger and larger deployments.

To do this, engineers will equip drones with cameras, and machine learning algorithms that make the drone’s operational and tactical decisions conditional on what these cameras show. As such, the defensive objective is still, as per usual, thwart the aggressor; **however, the aggressor is now the machine learning algorithms controlling the aggressor’s tech.**

For drones, what do these algorithms look like? How can they be thwarted? What risks and opportunities do they imply for the defense of critical infrastructure?

**While these questions might seem futuristic, that future is indeed approaching fast.**

## The supervised classification model

To begin the discussion, let’s look at the basic paradigm by which self-driving cars operate: the supervised classification model, which works as follows:

1. Engineers collect a large amount of “labeled data” in the form of `(cameras_images, what_the_driver_did)` pairs. This data is considered “labeled,” as each input, `cameras_images`, is coupled with an output, `what_the_driver_did`. An example of the former might be 360-degree images of a curving, barren road in Arizona; the latter, i.e. what the driver did in this moment in time, might be: depress the gas pedal by five degrees, and rotate the steering wheel counter-clockwise by seven.”
2. “Teach” a computer to, given photographic input, predict the correct driver action to be taken.

Roughly speaking, a self-piloting drone would not be different: it’d require equivalent labeled data from human-controlled flights, on which a supervised classification model would be trained.

## A likely progression of offensive drone tech

Weaponized drone deployment will likely progress as follows:

1. Deploy drones that pilot themselves; however, “finish” decisions are still executed by humans.
2. Deploy drones that pilot themselves; “finish” decisions are executed by the drone—when it is extremely confident in its choice.
3. Deploy swarms of drones that work together to more swiftly and efficiently achieve the aforementioned—attacking, exchanging data, self-sacrificing optimally, etc. throughout their attack.

**In a brave new world of machine-learned aggressors, what new opportunities for defense arise?**

## Adversarial examples in machine learning

Adversarial examples for supervised image classification models are inputs, i.e. images, whose pixels have been perturbed in a way imperceptible to a human eye, yet cause the classifier to change its prediction entirely.

An example[^1] looks as follows:

![png]({static}../images/saudi-adversarial/panda-adversarial-example.png)

Initially, the classifier is 57.7% confident that the image contains a panda; after adding noise, resulting in the image on the right—still, inarguably, a panda, to the human eye—the classifier has now changed its prediction to "gibbon" with 99.3% confidence.

**As such, knowledge of the data on which an aggressive drone was trained gives unique opportunity to algorithmically design adversarial examples, like the above, that confuse its classifier aboard.**

Concrete opportunities would include:

- **At “find”**: defensive lasers pointed at critical infrastructure dynamically perturb, by just a few pixels, its appearance, tricking the drone into thinking it has not found what it actually has.
- **At “fix”**: these lasers dynamically concoct sequences of perturbations, tricking the drone into thinking its target is moving erratically when it’s actually still.
- **At “finish”**: adversarial examples make the drone believe its strike would, put simply, cause more damage than it intends.

As adversarial examples are, once more, built from real examples, perturbed minimally, in a way imperceptible to the human eye, they are in fact difficult to, even a priori, teach our classifier to ignore. Said differently, well-designed adversarial examples are extremely effective.

## Out-of-distribution data

Machine learning algorithms are notoriously bad at “knowing what they don’t know.” Said differently, these algorithms only learn to make decisions about the type of data on which they’re trained. Were a classification model piloting a drone to be presented with an image of an ice-cream cone, it would, barring careful design, attempt to make a decision about this data all the same.

In a world of autonomous fly-and-finish drones, one would hope that its finish decisions are taken with extreme care. Fundamentally, this dovetails quickly into the notion of “out-of-distribution” data, i.e. data that the classifier knows it has not seen before, and about which it therefore neglects to make a prediction.

**As such, insight into the data on which an enemy’s system was trained naturally implies “defense by what’s different”**: show the drone images you know that it hasn’t seen before, and thereby increase its uncertainty around the decision at hand—buying time, and keeping your stuff in tact.

## Learning optimal defense via reinforcement learning

[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning), though exceedingly powerful, and often overhyped, is a relatively simple idea: given an environment and its state, try an action, observe a reward, and repeat the actions that give high rewards; additionally, periodically explore new actions you’ve never tried before just to see how it goes.

Reinforcement learning, or RL, requires vast amounts of data. As such, as a point of “meta defensive strategy”, trying out different types of adversarial-machine-learning attacks against a drone, seeing which work best, then repeating the best performers, while plausible in theory, would not work in practice if one drone approaches your one oil refinery but once a year.

This said, swarms of drones might change the game; *what if, in 20 years, Houthi militants deployed one autonomous, armed drone for every square mile of Saudi territory?* And what if we could simulate this scenario a priori, and learn how to optimally defend against the swarm?

To do this, you'd require:

- **A heavy investment into sensors**, on all of: the infrastructure you’re trying to protect, atmospheric conditions in which the drones are flying, tools to monitor the drones’ speeds and movements, etc. In other words, any and all technologies that capture the state of the environment, and the rewards, to as granular level of detail as possible.
- **Simulation environments.** In certain RL problems, this one potentially included, one has the delicious ability to generate data from which the algorithm can learn—by letting it play games with itself. [AlphaGo Zero](https://deepmind.com/blog/article/alphago-zero-starting-scratch) is one famous such example. Learning to optimally defend against a swarm of drones might fit the bit as well: deploy thousands of your own into a broad swath of desert, instruct them to “capture your flag”, then let your defensive systems get to work: taking actions, observing the subsequent reward—“Did I divert the drone away from my flag?”; “Did the drone hesitate more than usual before acting?”; etc.—and repeat those actions that work best.

## Summary

As the years roll forward, machine learning algorithms will continue to direct more and more of our most critical systems—financial markets, healthcare, transportation, and alas—the offensive technologies of political groups.

To wit, an understanding of the workings of these algorithms sheds light on the new risks these systems will introduce, and the new strategic opportunities that will thereafter arise.

## References

[^1]: Goodfellow, I., Shlens, J., Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples https://arxiv.org/abs/1412.6572
[^2]: (Thumbnail) “Adversarial AI: As New Attack Vector Opens, Researchers Aim to Defend Against It.” DataProtectionCenter.com - Tech and Security, 17 Apr. 2018, dataprotectioncenter.com/malware/adversarial-ai-as-new-attack-vector-opens-researchers-aim-to-defend-against-it/.
