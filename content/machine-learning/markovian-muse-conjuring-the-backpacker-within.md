Title: Markovian Muse: Conjuring the Backpacker Within
Date: 2015-06-29 00:17
Author: Will Wolf
Lang: en
Slug: markovian-muse-conjuring-the-backpacker-within
Status: published
Summary: Auto-generating travel blog posts via Markovian processes.
Image: images/backpacker.jpg

I haven't travel-blogged in a while - mainly because I haven't been traveling. When I was, the writing flowed easily: I'd wander, hike, try and fail, and humbling lessons about the world and its creatures seemed around every corner. There was plenty about which to write. Now, living here in New York City, and spending much of my time zealously science-ing data, I haven't found the same inspiration.

Piqued by Andrej Karpathy's [post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on auto-generating text with recurrent neural networks, and indeed wanting to write some more travel posts, I've decided to conjure the backpacker within. My tool of choice is a Markov chain - simple, and less black-box-y than an [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) - which will do the writing for us.

First thing's first: what is a Markov chain, and how does it work?

Markov chains are stochastic processes with many "states," or "nodes." At discrete time steps, the system may jump from one node to another with fixed probability. For example, assume our system is at State A at time $t = 0$. At $t = 1$, we jump to State B with probability $p = 0.4$, State C with probability $p = 0.3$, and remain at State A with probability $p = 0.3$. In addition, the system is "memoryless": the probability distribution ($A=0.3, B=.4, C=.3$ in the previous case) governing our move to the next state depends *only* on the current state, and none previous. 

I first learned about Markov chain's in undergrad as a way to model industrial processes. To auto-generate text with a Markov chain, the process is as follows:

1. Initialize an empty string, `s`.
2. Choose a word at random from a given body of text, and call it `word_1`. Call the subsequent word `word_2`.
3. Add `word_1` plus a whitespace to our string, such that `s += word_1 + ' '`.
4. Find all occurrences of `word_1 word_2` in the text. Then, create an array of each word that follows. For example, if `word_1 = "Will"` and `word_2 = "went"`, and our text contains the snippets `"Will went to"`, `"Will went further"`, and `"Will went north"`, then this array would be defined as `["to", "further", "north"]`. Duplicate elements should remain in the array.
5. Select a word at random from the array we just created and call it `word_3`. Then, set `word_1 = word_2` and `word_2 = word_3`.
6. Repeat steps 3-5 until the length of our string exceeds a user-defined string length.

The dataset I'm using is a full corpus of my [travel blog](http://willtravellife.com). Each post has its respective published-date attached, such that I can generate posts in the spirit of a certain timeframe. Let's give it a whirl!

Between January 1st, 2012 (the ~start of my journey), and March 1st, 2012, I was in Kenya, Tanzania, Uganda, Rwanda, United Arab Emirates, and Brazil. With these dates specified as start and end points respectively, and a maximum post length of 250 words, here's what we get:

> So--Ill Just Tell You What Ive Been Trying To Stitch
>
> Paths, and a chicken pie and a terribly anti-climactic pile of roughly 11 baked beans. Shortly after, I parted with the group for the kid who cant ride a manual motor scooter company arrived, yelled at a small valley, and see what was up around there. I really dont understand how 20 people could have stayed a day longer, but alas, the upcoming safari beckoned. This in the ocean at sunset, and meeting a Latvian Emirates stewardess whos going to talk about Africa. Africa is rather different from anywhere Ive ever eaten, and we arrive at Uhuru Peak just as the sun rise. 7:30AM: Eggs, toast, bananas, mangoes on the east coast. The touristy ones are Nungwe and Kendwa, each up north, and this beach resort. I get on the sill of the bags for a good Italian meal and a tuk-tuk with my camera. Light painting and zooming in and out during 10+ second exposures: Boom--artsy. The pinnacle of the sea, each poked with a Rwandese girl who is currently living in a beautiful city to boot. I arrive at Uhuru before sunrise, while it normally commands a fee of $49 stayed the next post. Kilimanjaro. Mount Kilimanjaro. F*cking beauty. We only stayed at the summit for about 15 minutes, and my world hasnt shattered nor universe exploded just yet. So--Ill just tell you what Ive been moving just a van, packed with roughly 20 seat-belt-less travelers, often outfitted with a toothpick, dipped in chili, and wrapped in newspaper.

Fast-forward a year. Between January 1st, 2013, and March 1st, 2013, I was in Senegal and Guinea. Using these dates and the same post length, we get:

> Stockpiled With Noodles, Cookies, And Overused Kitchenware-Wandered By 4 Or
>
> Wait a bit. You said this would work at noon. OK. Im going to inadvertently insult someone sooner or later. Its just going to be back in an hour. Maybe tomorrow. Im now hungry, so I know you arent fresh out; Im standing here with money-why dont you just pedal once, and then the owner grabbed both bottles from me and put them behind the bar-plastic tables and plastic chairs-and just no discernible superfluity, personal assumption, nor granular trace of coziness. Most are powered by generators, and once in a minute. 11:15am rolls around. Sir, whats the deal. It will be looked at, pointed at, and called names. I was 15 years old. I deposited $50 onto Full Tilt Poker, and I slept in (she was off work), went hiking, took some pictures, got a bit closer, to what this world is still alive and so very well, and a passing trucks. To be clear, its not exactly a boulder field that youre driving through, but after a cold, headlamp-lit bucket shower, I went through it all. Every country, every person, that time I stood looking up at the game they so love-the kids from the mothers and sisters finally, for the rest just looking to yell about something-formed in the air, at a fish market in Mauritania. 1592 National Pride in the car-only 3 seats still to fill. I read the first passing truck. The passengers were left to negociate a red-dirt road, with plenty of potholes, dodging cows!

So eloquent! Finally, let's stipulate the real start and end dates of my journey around the world: January 3rd, 2012 to March 12th, 2014. In all its glory:

> To Be A Travel Day Without Knowing Where Ill Ride
>
> Just wait a bit. Ill be in a wood-hut cafe, and am instantly hooked; Carey was a very natural combination of words like open, forward, transparent, and organic. Individual, on the front. I cant wait. Kitchen takeover is an equally innocent Palestinian local being mocked, harassed, and possibly the most serious lecture of all that stuff happened on this road, piste-section included, serves as a symbol of his back, uses the same people every day, challenge yourself, and from this guy boarded with a German guy who told me about another hour passing customs once in four months, when hiking in the above - all ruinous yet enchanting in their midst (this white guy (theres really not Superman. There is a language spoken by both the older and younger generations. If you dont, youre doing it yourself. The thing about traveling is an area of the house, through his carefully manicured fields, and into a friendship, a contact, and a dire lack of plumbing, food, running water, electricity, and general motley than the floor of the season. The weather in Patagonia is capricious at best, and terroristic at worst. Days like this place was an absolute nightmare of sog and wet. Follow the trail begins at the rink. Furthermore, the NHL guys were the only word I understood. As I said, its probably much more real. Emotionally, and certainly in big cities. 4. Study Blogs if Traversing Kazakhstan, Uzbekistan, and Tajikistan. My tips are based on my laptop at the.

Thanks for reading.

---
Code:

Code for this project can be found on my [GitHub](https://github.com/cavaunpeu/markovian-muse). It takes heavy inspiration from Greg Reda's [Nonsensical beer reviews via Markov chains](http://www.gregreda.com/2015/03/30/beer-review-markov-chains/), who does a terrific job with a similar topic.
