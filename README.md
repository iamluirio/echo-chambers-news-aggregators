# Study of Echo Chambers in News Media Aggregators
<div align="left">

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
<img width='70' height='30' src="https://github.com/user-attachments/assets/10c333d5-bf2a-43ff-9911-8bc067868045" />

</div>

The advent of social media and availability of news media platforms online has drastically transformed the news consumption behavior. Nowadays, the majority of the population tends to receive the daily news feeds online rather than the printed offline newspapers. These impertinent changes in news consumption demand news media sources to understand the pulse of the readers and adapt according to the trends; while personalization offers benefits, it also raises challenges, such as the risk of creating information bubbles or **echo chambers**. Existing research works define an *echo chamber* as an isolated virtual room or environment where the users’ beliefs and interests are reinforced by repeated exposure to information that aligns with user’s specific interests. 

There are several research works which focus on understanding and identifying echo chambers in different platforms, such as, blogs [Gilbert et al., 2009], forums [Edwards, 2013] and social media platforms [Zannettou et al., 2018]. However, these existing recent research works focus on identifying echo chambers on different social media platforms and there is no research work that focuses on echo chamber in news media aggregators. Existing works on social media platforms are not directly applicable for news media aggregators as there is no social relationship among users and users have only information about the news articles recommended to them. In this study we propose different measures that can quantitatively and qualitatively study characterization of echo chambers in news media aggregators across different users.

## Git repository 
- [Automated Collecting Data System](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Automated%20Collecting%20Data%20System): Collecting news for our study prove to be a time-intensive process, particularly when simulating the daily routines of multiple users engaging with diverse news articles. One significant time-consuming aspect is the meticulous attention to detail in simulating the entire user experience. We started creating an automated system for collecting news; we identify Selenium as a powerful open-source framework for automating web browsers. This system facilitate the simulation of user interactions
with news articles, automating tasks such as clicking buttons, navigating pages, and extracting data. Our system exploit **Selenium** library to simulate user login, providing a username and password corresponding to one of the simulated users. This allow us to seamlessly automate the process of accessing news articles significantly streamlining the data collection process for future iterations of our research.
  - [geckodriver.log](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Automated%20Collecting%20Data%20System/geckodriver.log): Log file generated by GeckoDriver, the WebDriver implementation for Mozilla Firefox. It provides detailed logging information about the interactions between Selenium and Firefox through GeckoDriver.
  - [newsMediaAggregators.ipynb](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Automated%20Collecting%20Data%20System/newsMediaAggregators.ipynb): Example of script that uses the NewsAPI to search for news headlines related to the username portion of the user's email address from three news sources (BBC News, CNN, and Fox News). You will need to replace "YOUR_API_KEY_HERE" with your own NewsAPI key to run this program.
  - [yahooNewsAggregator.ipynb](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Automated%20Collecting%20Data%20System/yahooNewsAggregator.ipynb): System that streamlines the process of gathering data, saving programmers from the job of compiling it manually. We identify Selenium as a powerful open-source framework for automating web browsers. Selenium, compatible with various web browsers. In this example, we provide the Yahoo News APIs with Mozilla Firefox implementation.

- [Content Analysis and Retrieval](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval): In this section, we analyze the actual content of the news presented to each user. By actual content, we mean a detailed analysis of the article text for each news article proposed to the user, employing various text analysis techniques.
  - [Single Modules](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules)
      - [Dependecy Tree Height](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules/Dependency%20Tree%20Height): Calculating the Dependency Tree Height of each sentence involves analyzing the syntactic structure of each sentence to determine how the words are related to each other.
      - [Filter Bubbles](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules/Filter%20Bubbles): Filter Bubbles refer to the phenomenon wherein individuals are increasingly exposed to information and perspectives that align with their existing beliefs, preferences, and interests, while being shielded from contradictory or diverse viewpoints. In the context of news media aggregators, such as online platforms and social media networks, algorithms curate and prioritize content based on user-specific data, including past behaviors, interactions, and demographic information. As a result, users are presented with a personalized stream of news and information that reinforces their preconceptions, limits exposure to alternative viewpoints, and may contribute to the polarization of societal discourse.
      - [Part-of-Speech (POS) Tagging](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules/Part-of-Speech%20(POS)%20Tagging): The Part-Of-Speech (POS) Tagging is a process in NLP that involves labeling each word in a text corpus with its corresponding part of speech, such as noun, verb, adjective, etc.
      - [Readability Analysis](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules/Readability%20Analysis): Readability Analysis is the evaluation of how easily a piece of text can be understood by readers. It involves assessing various linguistic and structural features of the text to determine its readability level. Factors such as sentence length, word complexity, and paragraph structure are considered in this analysis. Readability metrics, including formulas like the Flesch Reading Ease score, are often used to quantify these factors and provide a measure of how accessible the text is to readers. The goal is to ensure that written content is clear, comprehensible, and suitable for its target audience.
      - [Sentiment Analysis](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules/Sentiment%20Analysis): Sentiment Analysis is an NLP technique designed to determine the sentiment or emotional tone conveyed in a piece of text. The primary goal of Sentiment Analysis is to categorize the expressed opinions in the text as positive, negative, or neutral, providing valuable insights into the subjective viewpoints of individuals. This analytical process involves the use of algorithms and machine learning models to automatically analyse and interpret senti- ments expressed in various forms of textual data, such as social media posts, customer reviews, news articles.
      - [Text Pre-processing](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules/Text%20Pre-processing): It involves a series of analyses aimed at refining raw text data, transforming it from unstructured text strings into analyzable objects. This process begins with cleaning the text, which entails removing irrelevant elements such as punctuation, special characters, and stopwords. After cleaning, the text undergoes tokenization, breaking it down into individual words or tokens. Following tokenization, techniques like stemming or lemmatization may be applied to reduce words to their base or root form, thereby consolidating variations of words and enhancing analysis accuracy.
      - [Topic Modeling](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Content%20Analysis%20and%20Retrieval/Single%20Modules/Topic%20Modeling): Topic Modeling is technique to extract the hidden topics from large volumes of text. Topic model is a probabilistic model which contain information about the text. For example, a news paper corpus mcould contain topics like Economics, Sports, Politics, weather. Topic models are useful for purpose of document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection. Finding good topics depends on the quality of text processing, the choice of the topic modeling algorithm, the number of topics specified in the algorithm.

- [Insights and Data Patterns](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Insights%20and%20Data%20Patterns/Homophily%20between%20News%20Recomendation%20and%20News%20Consumption): In this section, we face a typology of analysis that carefully track how the news content changes over time. Our focus is on understanding how Google News evolves, including changes in topics shown in the For You section, and how news articles are arranged. We consider the factors that affect how Google News selects and displays content each day.
  - [Homophily between News Recomendation and News Consumption](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Insights%20and%20Data%20Patterns/Homophily%20between%20News%20Recomendation%20and%20News%20Consumption): Homophily is a concept that has been used to represent the tendency of individuals to form relationship with another individual who has similar interests. Several existing research works have proposed the utilization of homophily between users to understand the formation of echo chambers in social networks. However, as previously mentioned, there is no explicit relationship among users in news media aggregators. Therefore, in this section, we propose three different forms of homophily in news recommendation on the basis of the news consumption behavior.
    - [Homophily in Frequency](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Insights%20and%20Data%20Patterns/Homophily%20between%20News%20Recomendation%20and%20News%20Consumption/homophily_inFrequency.ipynb): We study the distribution of news articles recommended to a user based on their news consumption behavior. We study the number of news articles that are recommended to an user on a particular day that belongs to the user’s macro topic, micro topic and news which are neither macro nor micro (we refer to these as others).
    - [Temporal Homophily in News Recommendation](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Insights%20and%20Data%20Patterns/Homophily%20between%20News%20Recomendation%20and%20News%20Consumption/temporal_homophily.ipynb): We study whether the observed uniformity in news recommendation (from Homophily in Frequency) is maintained temporally. For our experiments, we consider a week and calculate frequency of news articles recommended to an user with respect to a particular news topic.
    - [Positional Homophily in News Recommendation](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Insights%20and%20Data%20Patterns/Homophily%20between%20News%20Recomendation%20and%20News%20Consumption/positional_homophily.ipynb): As Google News ranks the news articles based on relevance to the user, we explore the positional homophily in news recommendation for an user, i.e., we study the first position of any news article that belong to the user’s macro news topic.
  - [Case Study: Susceptibility to Propaganda News](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Insights%20and%20Data%20Patterns/Case%20Study%3A%20Susceptibility%20to%20Propaganda%20News): We delve into examining whether there exists a correlation between the user’s chosen news topics and their susceptibility to being recommended news with propaganda. Propaganda, in the context of news articles, refers to the dissemination of biased or misleading information intended to shape public opinion or promote a particular agenda. It often involves the use of persuasive techniques to influence individuals’ beliefs or attitudes, rather than presenting objective and unbiased reporting.
  - [New Users in Echo Chambers](): For a new user, Google News Recommender provides news based on the popularity of the news event with respect to the other news events. As observed, the quantity and selection of news presented to users are influenced
by various external factors. For instance, during a hypothetical election period for a political party, there may be a surge in the number of news articles related to politics compared to other topics. External events, seasons, or specific periods can significantly impact the news landscape, leading to variations in the content presented to users.
  - [User Similarity Analysis](): Our other analyses have mainly focused on individual users and how their interests were influenced by the news they read. We deliberately left out any outside factors that could affect their reading habits. In this section, instead of just looking at individual users, we want to compare them to see how similar they are. We consider lots of different factors to do this. This approach helps us retrieve patterns and trends that go beyond just one user. By looking at all these different factors, we hope to get a better understanding of how news changes during time.
  

