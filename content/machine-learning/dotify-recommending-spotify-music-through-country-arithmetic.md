Title: dotify: Recommending Spotify Music Through Country Arithmetic
Date: 2016-04-15 18:33
Author: Will Wolf
Lang: en
Slug: dotify-recommending-spotify-music-through-country-arithmetic
Status: published
Summary: A web app for Spotify music recommendation using [Implicit Matrix Factorization](http://yifanhu.net/PUB/cf.pdf) and "country arithmetic."
Image: images/dotify_screenshot.png

Ever since the release of [word2vec](https://en.wikipedia.org/wiki/Word2vec) I've been fascinated with embedding things - words, places, people - into vector space. Though not a mathematical historian, I don't believe this concept is at all new: matrix factorization methods like [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) have given us this ability for years. This said, one of the most exciting revelations of word2vec is the remarkably intuitive results of taking arithmetic combinations of these vectors - adding, subtracting, multiplying, etc. - with

$$vector_{king} - vector_{man} + vector_{girl} \approx vector_{queen}$$

being the classic party trick. With [dotify](http://dotify.herokuapp.com/), I set out to embed countries into vector space myself, and use these embeddings to recommend music. What are some songs like "Colombia" .. times "Turkey" .. minus "Germany," you ask?

![dotify Background]({static}/images/spotify_not_available.png)

To embed, I employ the timeless Implicit Matrix Factorization of [Hu, Koren, and Volinsky](http://yifanhu.net/PUB/cf.pdf) on daily Spotify Charts data. For a given song in a given country, implicit feedback is counted as the total number of times this song has been streamed throughout history. While the confidence matrix $C_{ui}$ is typically defined as $1 + \alpha(1 + R_{ui})$, I tweaked it to be $1 + \alpha\log{(1 + R_{ui})}$; this performed better empirically - likely due to the fact that many songs have stream counts orders of magnitude higher than others. I've been trying to implement more algorithms by hand as of late, which you'll find in the code. To use dotify: begin typing the name of a country, or simply press the down arrow, in order to select a country with your mouse or the enter key. Next, a dropdown of arithmetic operators will appear; select one in identical fashion. Feel free to keep entering countries and operators: your expression can be as long as you like. Finally, when you're ready to terminate your expression and receive song recommendations, simply enter the equals sign ("=") in the operator dropdown. For example:

![dotify Screenshot]({static}/images/dotify_screenshot.png)

The app is built with the following core technologies: Flask as both API endpoints and a web server; React and LESS on the front-end; Webpack for asset compilaton; Postgres database; Heroku for deployment.

Feedback is appreciated as always. Thanks for reading.

---
Code:

Code can be found on [GitHub](https://github.com/cavaunpeu/dotify), and dotify can be found [here](http://dotify.herokuapp.com/).
