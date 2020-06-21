---
layout: splash
title:  "Lobbyists4America"
permalink: /lobbyists4america/
date:   2019-06-21
--- 

# Lobbyists4America - Report

## Project description

The idea of my project is to develop a general understanding of the political landscape in the US
between the years 2008-2017. I will identify both relationships between politicians and attitudes of
politicians towards key topics. Apart from helping voters get a better idea of their politicians’
convicitions, it will support lobbyists in determining how to best allocate resources. Trying to get a
piece of legislation to pass, the lobbyist can e.g. use this analysis to identify which politicians need to
be swayed and which politicians have the largest influence on their colleagues.

This report is geared towards stakeholders.


### Questions
1. Basic analysis of tweet statistics:
Whose tweets are the most liked, most retweeted? Does the number of tweets correlate with the
number of likes and retweets? How does it correlate with the number of followers? Which correlation
is stronger?
2. Network analysis:
What are the relationships between politicians; are they amicable/hostile? What determines
these relationships? Are they mostly divided by party line? Which politicians have
the largest network, the biggest reach?
3. Topic modeling and sentiment analysis:
How do certain politicians feel about given topics? Are sentiments split along party line? Given
a topic, which politicians are the most opposed/in favor of it?

### Hypotheses
1. Exploratory data analysis shows that Donald Trump has the most followers, so I expect him to have
the most likes and reweets as well. I do expect the number of tweets to correlate with the number of
likes and retweets, however I expect the number of followers to be more decisive.
2. I expect relationships between politicians offer the same party to be amicable and more hostile
otherwards. I expect senators to
have the largest networks, mainly because they have usually been in office longer and because their
issues concern a larger amount of people.
3. I expect sentiments toward topics to be mostly split among party lines for big issues such as
health care and and immigration. 

### Approach

1. To answer question 1.) it will be enough to simply  compute
summary statistics. To filter out correlations it might be indicative to look at both the total number of
likes and the number of likes per tweet (same for number of retweets).
2. I will try to define relationships between politicians through mentions, replies and quotes. Sentiment
analysis can be used on mentions and replies to establish what kind of relationship two politicians have.
For simplicity I will assume that these relationships are undirected (meaning that if A likes B, B likes
A). As a metric we can then look at the total number of connections a politician has and further see how many of these connections are negative/positive.
3. Topics can both be extracted from the text body of a tweet and hashtags used in the tweet. I will assume that hashtags can be uniquely identified with a topic. Sentiment analysis can then be used to gauge a politicians opinian towards a topic.


## Results

### Tweet statistics 

![](/assets/img/lobbyists4america/nooffollowers.png)

Donald trump has the largest number of followers; in fact he has more followers than all Democrats and Republicans combined.

![](/assets/img/lobbyists4america/nooffollowers_byparty.png)


Cory Booker leads in number of tweets.

![](/assets/img/lobbyists4america/nooftweets.png)

However, in number of likes (aka favorites) per tweet he does not make it to the top 15 

![](/assets/img/lobbyists4america/favorite_count.png)

Looking at correlations within our data we see that the number of followers if correlated with the number of likes per tweet ($R^2 = 0.57$), however no correlation between number of followers and number of retweets per tweet can be observed ($R^2=0.01$). Using this knowledge we can fit a linear regression model:

$$ \left( \frac{\text{Likes}}{\text{Tweet}} \right) = a \cdot \left( \text{Followers} \right) + b $$

Using this model we can identify over- and underperfomers. Here are the top and bottom ten performers:

![](/assets/img/lobbyists4america/likes_bestandworst.png)

On a party level it seems like Democrats perform better on average than Republicans:

![](/assets/img/lobbyists4america/likes_compared_by_party.png)

Indeed a two-sample hypotheses test using non-parametric bootstrap reveals that democrats have on average 10 more likes per tweet as expected by their number of followers ($p < 10^{-5}$, 95% confidence interval: (6,13))

## Network analysis
I treated mentions, replies and quotes as (undirected) "Connections".
As predicted, Senators have on average 4.8 more connections than House Representatives and Governors ($p=7\cdot10^{-4}$, 95% CI: (2.5, 6.9)).

![](/assets/img/lobbyists4america/network_by_office.png)

Analyzing the sentiment of these connections (-1: most negative sentiment, +1 most positive sentiment), I find that inner-party relationships are slightly more positive than those across party lines (Average deviation: 0.054 (0.003,0.104), P-value: 0.0412)

![](/assets/img/lobbyists4america/party_sentiment.png)

From the network graph shown below we can infer several things. 
Senators (shown with bold borders) seem to cluster, which means that they form stronger networks with other Senators. Using the graph interactively we can identify which politicians communicate more across party lines. These are the ones closer to the boundary between Republicans and Democrats. We can further see that both Bernie Sanders and maybe more surprisingly Donald Trump is surrounded by mostly Democrats.

{% include network.html %}

## Topic modeling

For simplicity we only use hashtags to attach a topic to given tweet. 
Interestingly, only 1% of distinct hashtags make up about 50% of all usages of hashtags in this corpus:

![](/assets/img/lobbyists4america/tweetcomp.png)

Here is an overview of these hashtags:

![](/assets/img/lobbyists4america/wordcloud.png)

I picked "Obamacare/ACA" and "Immigration" for further analysis. It should be noted that sentiments reported should be taken with a grain of salt, as they gauge the tone of a given tweet and not necessarily the opinion of a politician towards a topic. For example "The best case against #Obamacare is proving to be Obamacare itself" is certainly coming from a politician opposed to the affordable care act but will be counted a having a positive sentiment because of the occurrence of the word "best". Therefore a more reliable measure may be to simply look about the tweet frequency about a given topic and split by keywords such as "repeal" for Obamacare and "reform" for immigration.

![](/assets/img/lobbyists4america/obamacare.png)

![](/assets/img/lobbyists4america/immigration.png)


# Conclusion 

I have identified which politicians have the largest platform, both among their colleagues and their followers. Moving forward, I would recommend Lobbyists4America focus on politicians who are both popular among their followers (many followers + more likes per tweet than expected) and have a strong network of fellow politicians. An investment into these key-players should pay off as their opinion towards a policy/topic will quickly propagate. Given the vast number of politicians included in this dataset I have often resorted to presenting only the top and bottom precentiles. In the future it might be helpful to have an interactive dashboard that stakeholders can use to browse/analyze the dataset themselves.


---
