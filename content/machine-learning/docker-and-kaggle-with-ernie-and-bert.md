Title: Docker and Kaggle with Ernie and Bert
Date: 2016-11-22 13:39
Author: Will Wolf
Lang: en
Slug: docker-and-kaggle-with-ernie-and-bert
Status: published
Summary: An introduction to what Docker is and why and how to use it for Kaggle.
Image: images/ernie_and_bert.png

This post is meant to serve as an introduction to what Docker is and why and how to use it for Kaggle. For simplicity, we will primarily speak about Sesame Street and cupcakes in lieu of computers and data.

One Monday morning, Ernie from the 'Street climbs out from under his red-and-blue pinstriped covers, puts both feet on the ground and opens his bedroom window. He stares out into a bustling metropolis of cookies and fur, straightens his banana-yellow turtleneck, lets out a deep, vigorous, crescent-shaped morning yawn and exclaims aloud: "Today, I'm going to make cupcakes for my dear friend Bert."

![ernie and bert]({static}/images/ernie_and_bert.png)

Unfortunately, Ernie has never made cupcakes before. But no matter! He darts hastily to the kitchen, pulls out a cookbook, organizes the ingredients and turns on his small Easy-Bake oven. "I'll experiment here. I'll make the greatest cupcake known to all stuffed-animal-kind. And when I'm happy with the result, I'll make 50 more," he shouts.

Hours later, Ernie's work is done: his cupcake - a 3-story stack of blueberry, strawberry and bacon-flavored sub-cakes - is the single best thing he's ever tasted. Far better than anything that fraud Cookie Monster had ever tried! Ernie is thrilled, and sits back in his now-filthy kitchen to admire the result. He thinks to Bert, and wonders how just quickly he can deliver his gift. "Now that I've baked the perfect cupcake, I'll just need to bake 50 more. This shouldn't be that hard. Right?"

Ernie spins around to look at this Easy-Bake. "Well, that thing only bakes one cupcake at a time. At that rate, 50 would take me days!" Spirits still high, he runs to the local bakery and asks to use their oven - this one much larger. They happily oblige, and Ernie starts baking right away.

Unfortunately, as he's mixing the ingredients he starts to have problems. The electric mixer breaks. The knife doesn't quite cut the strawberries in just the right way. The measuring cups have an ever-so-slightly different size. Ernie starts to stress. He thought he was at the finish line, but now realizes that he's really just at the start. While Ernie came equipped with the recipe to bake the cupcakes, he notes that he's now using all new tools in a completely new kitchen under completely different circumstances. "Can't a stuffed animal just bake a single cupcake in his small oven, then bring the recipe and ingredients to a bigger oven and bake a bunch more? Why does this need to be so complicated?"

## Enter Docker

In sadness and despair, Ernie wanders to the seaport to clear his mind. There, he comes across hundreds of blue and white, SUV-sized shipping containers and gets a funny idea: "What if I did my baking in there? I'll move all of my tools inside - the cutting board, the knife, the mixer, the utensils - and write the recipe on the inner wall. The only thing missing will be the oven, but I can get that anywhere. That way, using the oven *chez moi,* I can continue to bake one cupcake at a time; conversely, using the oven at the bakery I can bake a whole lot more. Perfect. Ernie grabs the first container he sees and races home to pack it full.

After writing the ingredients on the container's inner wall, Ernie realizes that if he's going to bring this container to the bakery, it better be light. If not, he won't be able to carry it! Therefore, instead of actually including his tools - the knife, the mixer, etc. - he simply writes down the names and numbers of these products and instructions as to where they can be acquired. Similarly, instead of including the actual ingredients for the cupcakes, he expects them to be available at the bakery itself. Then, when the recipe says "take 3 tablespoons of sugar from the cupboard," that sugar will have already been placed in the cupboard itself.

## Enter Kaggle

Baking cupcakes on Sesame Street is a metaphor for building models for Kaggle. Typically, we build small prototypes on our local machine, then temporarily rent a more powerful machine sitting on a farm somewhere in Virginia to do the heavy lifting. In Kaggle competitions, Ernie's initial problem is all too common: even after finding an electric mixer, measuring cups, etc. comparable to his own - i.e. even after installing all those libraries on our remote machine that we had on our local - the environments still weren't quite the same and problems therein arose. Docker containers solve this problem: if we can bake our cupcake once in our kitchen, we can deterministically re-bake it *n* times in any kitchen - and preferably one with an oven much more powerful than our own.

## Enter remote instances

A remote instance is the bakery: it is a computer, like ours, that can process data faster and in larger quantities. In other words, it is a kitchen with a much bigger oven.

## Cooking utensils and ingredients

In lieu of including cooking utensils in our container we merely specify which utensils we need and how to acquire them. For a Kaggle competition, this is akin to installing the libraries - pandas, scikit-learn, etc. - necessary for the task at hand. Once more, we do not include these libraries in our container, but instead provide instructions as to where and how to install them. In practice, this often looks like a `pip install -r requirements.txt` in our [`Dockerfile`](https://docs.docker.com/engine/reference/builder/).

In lieu of including ingredients in our container we merely assume they'll be available in our host bakery. This is a bit trickier than it sounds for the following reasons:

1. Our host bakery is several blocks from our home. If we want ingredients to be available in that bakery, we're going to need to physically carry them there in some sense. 2. Even after physically bringing ingredients to the bakery, they still won't be immediately available inside the container. Remember, after bringing our container to the bakery, the cooking that transpires within the container is isolated from the rest of the bakery itself; it interfaces only with the bakery's oven.

For a Kaggle competition, how do we make local data available *within the container*, *on a remote machine?*

### Docker Volumes

[Docker Volumes](https://boxboat.com/2016/06/18/docker-data-containers-and-named-volumes/) allow data to be shared between a directory inside of a container and a directory in the local file system of the machine hosting that container. This is akin to Ernie:

1. Carrying his ingredients to the bakery, along with (but not inside) his container.
2. Upon arrival, placing a jar of sugar in a blue bucket in the corner of the room.
3. Stipulating that, upon beginning to bake inside of the container at the bakery, ingredients should be shared between the blue bucket in the corner of the room and the cupboard. That way, when the recipe says "get a jar of sugar from the cupboard," Ernie can reach into the cupboard inside of the container and retrieve the jar of sugar *from the blue bucket sitting in the corner of the bakery.* Remember: the container did not ship with any ingredients inside; the cupboard, therefore, would have itself been empty.

Carrying the container to the bakery is akin to a simple `docker run` onto the remote machine. Carrying ingredients to the bakery, i.e. placing a data file on the local file system of the remote machine, is much less sexy. In the simplest sense, this is akin to using `scp` or `rsync` to transfer a file from the local machine to the remote, or even using `curl` to download a file directly onto the remote machine itself.

In practice, this often looks like:

```
docker
    --tlsverify
    --tlscacert="$HOME/.docker/machine/certs/ca.pem"
    --tlscert="$HOME/.docker/machine/certs/cert.pem"
    --tlskey="$HOME/.docker/machine/certs/key.pem" -H=tcp://12.34.56:78
run
    --rm
    -i
    -v
    /data:/data kaggle-contest build_model.sh
```

## Cooking tools that you can't buy at the store

To bake his cupcake, Ernie used a one-of-a-kind cutting board that Bert had hand-molded for him. How can he use this at the bakery? In Kaggle terms: how can I use a library in my project that is not available on a public package repository (i.e. one that I built myself)?

To this end, there's really no secret sauce. With the cutting board/library, we can either:

1. Include it in our container and deal with the extra weight.
2. Treat it as an ingredient, carry it to the bakery, and access it via a Docker Volume.

## Happy cooking

Moving your local development inside of a Docker container, and/or Dockerizing this local environment once you're ready to use a remote resource to do the heavier lifting, will ensure you only have to figure out how to bake the cupcake once. Prototype locally, then send stress-free to the bakery for mass production.

Happy cooking.

---
Additional resources:

Here's two resources I found very helpful when learning about Docker for Kaggle:

1.  [Workflow, Serialization & Docker for Kaggle](https://speakerdeck.com/smly/workflow-serialization-and-docker-for-kaggle)
2.  [How to get started with data science in containers](http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/)
