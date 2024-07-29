# Study of Echo Chambers in News Media Aggregators
<div align="left">

<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/nltk-grey?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/spacy-grey?style=for-the-badge&logo=spacy&logoColor=lightblue" />
<img src="https://img.shields.io/badge/selenium-grey?style=for-the-badge&logo=selenium&logoColor=green" />
</div>

The advent of social media and availability of news media platforms online has drastically transformed the news consumption behavior. Nowadays, the majority of the population tends to receive the daily news feeds online rather than the printed offline newspapers. These impertinent changes in news consumption demand news media sources to understand the pulse of the readers and adapt according to the trends; while personalization offers benefits, it also raises challenges, such as the risk of creating information bubbles or **_echo chambers_**. Existing research works define an *echo chamber* as an isolated virtual room or environment where the users’ beliefs and interests are reinforced by repeated exposure to information that aligns with user’s specific interests. 

There are several research works which focus on understanding and identifying echo chambers in different platforms, such as, blogs [[Gilbert et al., 2009]](https://ieeexplore.ieee.org/document/4755503), forums [[Edwards, 2013]](https://philpapers.org/rec/EDWHDP) and social media platforms [[Zannettou et al., 2018]](https://arxiv.org/abs/1802.05287#:~:text=In%20August%202016%2C%20a%20new,suspended%20from%20other%20social%20networks.). However, these existing recent research works focus on identifying echo chambers on different social media platforms and there is no research work that focuses on echo chamber in news media aggregators. **Existing works on social media platforms are not directly applicable for news media aggregators as there is no social relationship among users and users have only information about the news articles recommended to them**. In this study, we propose different measures that can quantitatively and qualitatively study characterization of **echo chambers in news media aggregators across different users**.

## Introduction
Below, we give some preliminary information about the study elements that we used for our research, and how the tools exposed were used consequently. We want to provide a context so that the user can use the tools for similar purposes.

For a 360° overview of the project in detail, I recommend the reader to read my personal master's thesis, or to read the subsequently published paper about the project.

- **News Media Aggregator**: A _news media aggregator_ is a digital platform or service that collects and compiles news articles, updates, and content from various sources across the internet into a single location. These aggregators use algorithms to organize and present the information to users based on their preferences and interests. In this study, we underscore the significance of providing users with a personalized space that automatically populates with news articles relevant to their interests; by examining the content selection and presentation practices of news aggregators, we aim to uncover any patterns of imbalance or favoritism towards specific viewpoints, ideologies, or sources.
- **News Article**: An integral component of the subsequent analyses, as well as a fundamental parameter of this study, revolves around how the _news articles_ are presented by the news media aggregator. We closely examine different aspects
of the articles, looking at both how they’re written and what they’re about.
- **Political Biases in News Media**: Political topic news are presented every day by any newspaper and any aggregation site: they inform citizens about the actions and decisions of their governments, helping them make informed choices during elections and hold their leaders accountable for their actions. Political bias in news articles can contribute to the formation of echo chambers: when news consistently present information from a particular political perspective, they reinforce the beliefs of their audience while potentially ignoring or downplaying alternative viewpoints. 

For our type of study, we chose to apply the search tools and indexes to news articles obtained from **Google News**[^important1] news media aggregator. In particular, we refer to the evolution of the **For You section**: the personal area dedicated to the user that evolves over time based on his interests.

To study real life users news consumption behavior without any inherent bias, we carefully design our simulation such that we have users with different news topical interests who belong to different locations. For our experiments, we conduct user profile creation task from **USA and India**[^important2] countries.

We assign each user two main news topics. The first topic, called **majority topic**, is the one that **the user is most interested in**. This means that **they read more news articles about this topic compared to the second assigned topic**. On the other hand, the second assigned topic is referred to as the **minority topic**. Even though users read news about both topics every day, they read fewer articles about the minority topic compared to the majority one. Therefore, we consider users with three different types of political leaning: **Republican political leaning, Democratic political leaning and Neutral political leaning, for USA**. Subsequently, **for India, we consider the political leaning as Government leaning, Opposition leaning and Neutral political leaning**[^important3] . For non-political topics, we consider different news topic interests, such as, **Sports, Entertainment, Technology, World and Business**. Although there can be other news topics, we select these topics on the basis of the popularity of the topics in Google News platform.

When a user is assigned to a particular political party, it means **they not only read news related to that specific party, but also news that favors that**. This ensures that the user is exposed solely to news content aligned with their political affiliation, thus creating a personalized history consisting exclusively of news favorable to that specific party. This approach is used for each assigned topic.

[^important1]: <small>Our selection of a particular news media aggregator for this analysis is not intended as a personal critique or an assertion of bias on the part of the aggregator. Our goal is not to demonstrate or confirm any predisposition towards a specific political party, nor to suggest a preference for certain topics from one newspaper over another. The choice of this aggregator is purely based on its prominence and widespread daily use among the majority of users. This ensures that our study is relevant and reflective of the general news consumption habits of a large audience.</small>

[^important2]: <small>The selection of these two nations for our analysis is based on their significant geopolitical influence and physical size. This choice is not intended to imply that these nations exhibit more or less bias compared to other states or nations. Instead, their prominence on the global stage and the breadth of their media landscapes make them suitable subjects for a comprehensive study of echo chambers in news media.</small>

[^important3]: <small>The team members involved in this project are students who are pursuing or have completed a bachelor's or master's degree in Computer Science. Their ability to identify potential biases within news articles, particularly in relation to political parties, has been developed through preliminary training focused on the key aspects of US and Indian politics. Moreover, the project leader brings substantial expertise to the team, having engaged in numerous studies related to the analysis of fake news and bias, including those of a political nature. This combination of academic background ensures a well-rounded and informed approach to the evaluation of news media content in this study.</small>

| Acronym                  | Explanation                                   |
|--------------------------|-----------------------------------------------|
| _U<sub>n</sub>_          | It refers to the n-th user from USA.          |
| _I<sub>n</sub>_          | It refers to the n-th user from India.        |
| _N<sub>n</sub>_          | It refers to the n-th new user (used only for one specific analysis). |
| _rep_                    | Republican Party.                             |
| _dem_                    | Democratic Party.                             |
| _unbiased_ / _neu_       | Neutral user. It means that he does not read news regarding the favoring of a particular political party. |
| _macro_                  | Majority topic assigned to the user.          |
| _micro_                  | Minority topic assigned to the user.          |
| _others_                 | News which are neither macro nor micro.       |
| _Home_                   | It refers to the Google News articles' homepage. |
    
                                                                  

## Usage
```
git clone https://github.com/iamluirio/echo-chambers-news-aggregators.git
```

We show the code to be readily utilized for testing purposes, allowing you to explore the presence of echo chambers based on our analysis within a defined text-containing system. This setup also enables you to experiment with the tools we employed throughout our study; this will aid in replicating our methodology and verifying the robustness of our findings in your own investigations.

## Insights and Data Patterns
In this section, we face a typology of of analysis that carefully track how the news content changes over time. Our focus is on understanding how Google News evolves, including changes in topics shown in the For You section, and how news articles are arranged.

### Homophily between News Recomendation and News Consumption
Homophily is a concept that has been used to represent the tendency of individuals to form relationship with another individual who has similar interests. Several existing research works have proposed the utilization of homophily between users to understand the formation of echo chambers in social networks. 

However, as previously mentioned, there is no explicit relationship among users in news media aggregators. Therefore, in this section, we propose three different forms of homophily in news recommendation on the basis of the news consumption behavior.

#### Homophily in Frequency
We study the distribution of news articles recommended to a user based on their news consumption behavior. We study the number of news articles that are recommended to an user on a particular day that belongs to the user’s macro topic, micro topic and news which are neither macro nor micro (we refer to these as others). Usually, the nuts belong to the others section, naming the Others section, to political news (even for users who do not follow political topics), and news belonging to the Google News Homepage, i.e. popular and global news.

We opt for a specific day and replicate this type of analysis across several days. Our focus center on the top 20 news articles featured in the For You section of each user. This approach allow us to systematically assess and compare the evolution of news content over the selected timeframe. Our observations indicate that the news recommendation varies on the basis of the topics a user follows as macro and micro topic.

For example, ```REP = [3, 1, 2, 2]```  means that, for the specific day of analysis, _U<sub>1</sub>_ got three news articles about republican party in his For You page, _U<sub>2</sub>_ one news article about republican party, and so on.

```python
# Set data for the plot
REP = [3, 1, 2, 2]     
DEM = [6, 5, 8, 4]
UNB = [0, 0, 3, 7]
MIC = [2, 5, 1, 3]
OTH = [9, 9, 6, 4]
x = np.arange(len(REP))  # x-axis values

# Set bar width
bar_width = 0.15

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))  # set the figure size to 12x8
colormap = plt.get_cmap('Blues')  # Choose the colormap
colors = [colormap(i) for i in np.linspace(0.2, 1, 5)][::-1]  # Generate 5 colors from the colormap starting at 0.2 and reverse the order

ax.bar(x - 2.0 * bar_width, REP, bar_width, label='rep', color=colors[0])
ax.bar(x - 1.0 * bar_width, DEM, bar_width, label='dem', color=colors[1])
ax.bar(x, UNB, bar_width, label='unbiased', color=colors[2])
ax.bar(x + 1.0 * bar_width, MIC, bar_width, label='micro', color=colors[3])
ax.bar(x + 2.0 * bar_width, OTH, bar_width, label='others', color=colors[4])

# Set the x-axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(['U1', 'U2', 'U3', 'U4'], fontsize=16)  # Set x-axis label font size
ax.set_xlabel('Users', fontsize=20)

# Set y-axis label and font size
ax.set_ylabel('Num of News Articles', fontsize=20)

# Increase the y-axis tick labels font size
ax.tick_params(axis='y', labelsize=16)

# Add a legend and title
ax.legend(loc='upper right', fontsize=12)  # Adjust legend location to upper right

# Show the plot
plt.show()
```

<div align="center">
    <img src="https://github.com/user-attachments/assets/6b03e860-8b09-4e8d-b3a8-1def906a4f7a" />
</div>



