Title: Travel Recommendations with Jaccard Similarities
Date: 2015-10-03 02:40
Author: Will Wolf
Lang: en
Slug: travel-recommendations-with-jaccard-similarities
Status: published
Summary: A Scala-based web app that gives travel recommendations via Twitter data and the Jaccard similarity.
Image: figures/colombia_recommendations.png

I recently finished building a [web app](http://countryrecommender.herokuapp.com/) that recommends travel destinations. You input a country, and it provides you with 5 other countries which you might also enjoy. The recommendations are generated via a basic application of [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering). In effect, you query for a country, and the engine suggests additional countries enjoyed by other users of similar taste. The methodology is pretty simple:

1. Capture travel-related tweets from the [Twitter API](http://twitter4j.org/en/), defined as any containing the strings "#ttot", "#travel", "#lp", "vacation", "holiday", "wanderlust", "viajar", "voyager", "destination", "tbex", or "tourism."
2. If a tweet contains the name of one of 248 countries or territories, or their respective capitals, label this tweet with this country. For example, the tweet "backpacking Iran is awesome! #travel" would be labled with "Iran."
3. Represent each country as a set of the user_id's who have tweeted about it.
4. Compute a [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index) - defined as the size of the intersection of 2 sets divided by the size of their union - between all combinations of countries.
5. When a country is queried, return the 5 countries Jaccard-most similar. The length of the bars on the plot are the respective similarity scores. So - let's try a few out!

![colombia recommendations]({filename}/figures/colombia_recommendations.png)

Not bad. Venezuela - neighbor to the East - is recommended most highly. Again, this implies that those tweeting about Colombia were also tweeting about Venezuela.

![malaysia recommendations]({filename}/figures/malaysia_recommendations.png)

Seems logical.

![india recommendations]({filename}/figures/india_recommendations.png)

Strange one, maybe? Then again, all countries - especially Greece, Italy, and France - are universally popular travel destinations, just like our query country India. As such, it's certainly conceivable that they were being promoted by similar users. We'd want to read the tweets to be sure.

Moving forward, I plan to implement a more robust similarity metric/methodology. While a Jaccard similarity is effective, it's a little bit "bulky": most notably, being a *set* similarity metric, it doesn't consider repeat tweets about a given country. For example, if User A tweeted 12 times about Hungary and 1 time about Turkey, a Jaccard similarity would consider these behaviors equal (by simply including User A in each country's respective set). As such, a cosine-based metric might be more appropriate. See [here](http://www.benfrederickson.com/distance-metrics/) for an excellent overview of possible approaches to a "people who like this also like" analysis.

Once more, the app is linked [here](http://countryrecommender.herokuapp.com/). In addition, the code powering the app can be found on my [GitHub](https://github.com/cavaunpeu/countryrecommender). Backend in Scala, and front-end in HTML, CSS, and native JavaScript. Do take it for a spin, and [let me know](https://twitter.com/WillTravelLife) what you think!
