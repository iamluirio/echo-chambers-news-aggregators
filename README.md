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
- **News Media Aggregator**: A _news media aggregator_ is a digital platform or service that collects and compiles news articles, updates, and content from various sources across the internet into a single location. These aggregators use algorithms to organize and present the information to users based on their preferences and interests. In this study, we underscore the significance of providing users with a personalized space that automatically populates with news articles relevant to their interests; by examining the content selection and presentation practices of news aggregators, we aim to uncover any patterns of imbalance or favoritism towards specific viewpoints, ideologies, or sources.
- **News Article**: An integral component of the subsequent analyses, as well as a fundamental parameter of this study, revolves around how the _news articles_ are presented by the news media aggregator. We closely examine different aspects
of the articles, looking at both how they’re written and what they’re about.
- **Political Biases in News Media**: Political topic news are presented every day by any newspaper and any aggregation site: they inform citizens about the actions and decisions of their governments, helping them make informed choices during elections and hold their leaders accountable for their actions. Political bias in news articles can contribute to the formation of echo chambers: when news consistently present information from a particular political perspective, they reinforce the beliefs of their audience while potentially ignoring or downplaying alternative viewpoints. 

For our type of study, we chose to apply the search tools and indexes to news articles obtained from **Google News**[^important] news media aggregator. 

To study real life users news consumption behavior without any inherent bias, we carefully design our simulation such that we have users with different news topical interests who belong to different locations. For our experiments, we conduct user profile creation task from **USA and India** countries. To ensure a comprehensive representation of user demographics and preferences, we divide the user creation process among the team members working on the project; this approach allows us to capture diverse perspectives and insights from individuals hailing from distinct cultural backgrounds and geographical regions.

Therefore, we consider users with three different types of political leaning: **Republican political leaning, Democratic political leaning and Neutral political leaning, for USA**. Subsequently, **for India, we consider the political leaning as Government leaning, Opposition leaning and Neutral political leaning**. For non-political topics, we consider different news topic interests, such as, **Sports, Entertainment, Technology, World and Business**. Although there can
be other news topics, we select these topics on the basis of the popularity of the topics in Google News platform.

[^important]: <small>Our selection of a particular news media aggregator for this analysis is not intended as a personal critique or an assertion of bias on the part of the aggregator. Our goal is not to demonstrate or confirm any predisposition towards a specific political party, nor to suggest a preference for certain topics from one newspaper over another. The choice of this aggregator is purely based on its prominence and widespread daily use among the majority of users. This ensures that our study is relevant and reflective of the general news consumption habits of a large audience.</small>

## Usage
```
git clone https://github.com/ptrespidi/echo-chambers-news-aggregators.git
```


We show the code to be readily utilized for testing purposes, allowing you to explore the presence of echo chambers based on our analysis within a defined text-containing system. This setup also enables you to experiment with the tools we employed throughout our study; this will aid in replicating our methodology and verifying the robustness of our findings in your own investigations.



