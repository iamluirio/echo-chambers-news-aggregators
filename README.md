# Study of Echo Chambers in News Media Aggregators
<div align="left">

<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/transformers-grey?style=for-the-badge&logo=huggingface&logoColor=yellow" />
<img src="https://img.shields.io/badge/nltk-grey?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/spacy-grey?style=for-the-badge&logo=spacy&logoColor=lightblue" />
<img src="https://img.shields.io/badge/selenium-grey?style=for-the-badge&logo=selenium&logoColor=green" />
</div>

The advent of social media and availability of news media platforms online has drastically transformed the news consumption behavior. Nowadays, the majority of the population tends to receive the daily news feeds online rather than the printed offline newspapers. These impertinent changes in news consumption demand news media sources to understand the pulse of the readers and adapt according to the trends; while personalization offers benefits, it also raises challenges, such as the risk of creating information bubbles or **_echo chambers_**. Existing research works define an *echo chamber* as an isolated virtual room or environment where the users’ beliefs and interests are reinforced by repeated exposure to information that aligns with user’s specific interests. 

There are several research works which focus on understanding and identifying echo chambers in different platforms, such as, blogs [[Gilbert et al., 2009]](https://ieeexplore.ieee.org/document/4755503), forums [[Edwards, 2013]](https://philpapers.org/rec/EDWHDP) and social media platforms [[Zannettou et al., 2018]](https://arxiv.org/abs/1802.05287#:~:text=In%20August%202016%2C%20a%20new,suspended%20from%20other%20social%20networks.). However, these existing recent research works focus on identifying echo chambers on different social media platforms and there is no research work that focuses on echo chamber in news media aggregators. **Existing works on social media platforms are not directly applicable for news media aggregators as there is no social relationship among users and users have only information about the news articles recommended to them**. In this study, we propose different measures that can quantitatively and qualitatively study characterization of **echo chambers in news media aggregators across different users**.

## Important Considerations
Before continuing, we recommend paying close attention to the small notes at the bottom of the page. We would like to emphasize that the chronological snapshot of the start of 2023 in politics reveals a period marked by significant events that undoubtedly influenced the dynamics of our study. It is important to recognize that every analysis involving experiments and the collection of data from temporal sources inherently carries a specific temporal timestamp. This timestamp denotes the period during which the project is conducted and the data is gathered.

The temporal aspect of our study is not merely a logistical consideration; it profoundly influences the nature and interpretation of the data collected. As a result, the data collected during this period may inherently carry a bias, reflecting the heightened emotions, polarization, and discourse surrounding the impeachment proceedings. Additionally, it is crucial to note that we do not intend to personally judge whether the news aggregator favors the creation of echo chambers. Instead, we present our experiments and analyses, allowing the reader to draw their own conclusions based on the evidence provided.

## Table of Content
- [Introduction](#introduction)
- **[Insights and Data Patterns](#insights-and-data-patterns)**
  - [Homophily between News Recomendation and News Consumption](#homophily-between-news-recommendation-and-news-consumption)
    - [Homophily in Frequency](#homophily-in-frequency)
    - [Temporal Homophily in News Recommendation](#temporal-homophily-in-news-recommendation)
    - [Positional Homophily in News Recommendation](#positional-homophily-in-news-recommendation)
  - [User Similarity Analysis](#user-similarity-analysis)
    - [User Viewpoint Similarity Index](#user-viewpoint-similarity-index)
    - [User Viewpoint Representativeness Score](#user-viewpoint-representativeness-score)
    - [User Stance Index](#user-stance-index)
  - [New Users in Echo Chambers](#new-users-in-echo-chambers)
  - [Case Study: Susceptibility to Propaganda News](#case-study-susceptibility-to-propaganda-news)
- **[Content Analysis and Retrieval](#content-analysis-and-retrieval)**
  - [Extracting Urls from Dataset](#extracting-urls-from-dataset)
  - [Text Pre-processing Overview](#text-pre-processing-overview)
    - [Stemming Analysis](#stemming-analysis)
    - [Lemmatization](#lemmatization)
  - [Text Analytics Techniques](#text-analytics-techniques)
    - [Sentiment Analysis](#sentiment-analysis)
    - [Readability Analysis](#readability-analysis)
    - [Part-Of-Speech (POS) Tagging](#part-of-speech-(pos)-tagging)
    - [Topic Modeling](#topic-modeling)
    - [Dependency Tree Height](#dependency-tree-height)
  - [Filter Bubbles](#filter-bubbles)
    - [Average News Document Stance](#average-news-document-stance)
    - [Entropy User Score](#entropy-user-score)
- **[Automated Collecting Data System](#automated-collecting-data-system)**

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
                                                            
## Insights and Data Patterns
In this section, we face a typology of of analysis that carefully track how the news content changes over time. Our focus is on understanding how Google News evolves, including changes in topics shown in the For You section, and how news articles are arranged.

### Homophily between News Recomendation and News Consumption
**Homophily** is a concept that has been used to represent **the tendency of individuals to form relationship with another individual who has similar interests**. Several existing research works have proposed the utilization of homophily between users to understand the formation of echo chambers in social networks. 

However, as previously mentioned, **there is no explicit relationship among users in news media aggregators**. Therefore, in this section, **we propose three different forms of homophily in news recommendation on the basis of the news consumption behavior**.

#### Homophily in Frequency
We study **the distribution of news articles recommended to a user based on their news consumption behavior**. We study **the number of news articles that are recommended to an user on a particular day that belongs to the user’s macro topic, micro topic and others**. Usually, the nuts belong to the others section, to political news (even for users who do not follow political topics), and news belonging to the Google News Homepage, i.e. popular and global news.

We opt for a specific day and replicate this type of analysis across several days. Our focus center on the top 20 news articles featured in the For You section of each user. This approach allow us to systematically assess and compare the evolution of news content over the selected timeframe. Our observations indicate that the news recommendation varies on the basis of the topics a user follows as macro and micro topic.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/6b03e860-8b09-4e8d-b3a8-1def906a4f7a" style="width: 45%; margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/96da1851-08c5-4c6f-84b7-2aa0cf557dc8" style="width: 45%;" />
    </div>
    <p><em>Figure 1: Visualizing the Ratio of News Recommended to Politic Users as Macro and belong to USA.</em></p>
</div>

Our observations indicate that the news recommendation varies on the basis of the topics a user follows as macro and micro topic. For example, we observe users with pro-Government leaning get news related to pro- Government leaning more than the news related to Opposition leaning. We observe similar phenomenon for users who have pro-Opposition leaning for news related to pro-Opposition leaning. However, we do not observe the same with respect to users from USA. While users who follow democratic news gets a higher proportion of news related to democratic rather than republican news, users who follow Republican Party as their macro topic do not always get more number of pro-Republican news than pro-Democratic news.

#### Temporal Homophily in News Recommendation
Currently, we study whether **the observed uniformity in news recommendation is maintained temporally**. For our experiments, we consider a week and **calculate frequency of news articles recommended to an user with respect to a particular news topic**. For example, for the news topic politics, we calculate the number of news articles related to pro-Government recommended to an user who has politics as major news topic. We repeat this subsequently for pro- Government and pro-Opposition leaning for users from India, pro-Republican and pro-Democratic for users from USA, respectively.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/86047e26-10ce-4d5a-ab63-df75d9f2da8c" style="width: 45%; margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/1129d6ae-db0e-407b-8106-cc0887d4e2a1" style="width: 45%;" />
    </div>
    <p><em>Figure 2: Pro-Government Temporal Pattern (left) and Pro-Opposition Temporal Pattern (right). </em></p>
</div>

Our observations indicate that the number of news articles recommended to an user which belongs to her macro news topic is much higher than any other news topic irrespective of the day of the week, date or location of the user. We observe similar behavior irrespective of the macro news topic being political or non-political. However, for users with pro-Republican political leaning, they are either recommended news which cover republican news or pro-Democratic view point. We observe this similar pattern through different experiments for pro-Republican users.

#### Positional Homophily in News Recommendation
As Google News ranks the news articles based on relevance to the user, **we explore the positional homophily in news recommendation for an user**, i.e., we study **the first position of any news article that belong to the user’s macro news topic**.
An user has high positional homophily in news recommendation if the news articles which match her macro news topic is ranked early in her news feed. For our experiment, we repeat this for all the users of both USA and India for 5 days, respectively.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/a80b4dbd-195f-4e4e-9374-4a1cf3b1a681" style="width: 45%; margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/f0403a02-ddca-4d5d-8062-5ed94c6adc8d" style="width: 45%;" />
    </div>
    <p><em>Figure 3: Positional Homophily in News Recommendation.</em></p>
</div>

Our observations indicate that irrespective of the macro news topic, the position of the first news ranges between 1-3 mostly. Additionally, we observe that for an user, the position of their macro news topic rarely appears beyond position 3 in the Google News page. This further highlights that the news recommended to different users are ranked differently on any day irrespective of the news events of that day and is dependent only on the user’s news topic choice and political leaning.

### User Similarity Analysis
Until now, our analyses have mainly focused on individual users and how their interests were influenced by the news they read. Now, **we want to compare them to see how similar they are**. This approach helps us retrieve patterns and trends that go beyond just one user. By looking at all these different factors, we hope to get a better understanding of how news changes during time. 

To study the variance in news recommendation among users quantitatively, we study 3 different measures, namely, **_User Viewpoint Similarity Index_**, **_User Viewpoint Representation Index_** and **_User Stance Index_**. Through these metrics and observations, we intend to capture how the news recommendation is similar between a pair of users on the basis of their macro, micro news topic and political leaning.

#### User Viewpoint Similarity Index
For this type of analysis, **we compare similarity in the news recommended between a pair of users on the basis of the topics of the news**. We consider the topics as a combination of all the possible macro and micro news topics.

Suppose we approach the situation from the perspective of users from India; the first example we report concerns the situation of the topics according to the point of view of users from India, i.e., pro- Government, pro-Opposition, Neutral, Sports, Entertainment, Technology, World and Business, for a total of 8 topics. Therefore, **for an user _I<sub>j</sub>_ , _Topic Distribution_ is a vector of size 8 where each position of the vector represents each topic, and the value is the frequency of the news articles recommended to _I<sub>j</sub>_ on that topic**. 

Suppose we take two users: user _I<sub>1</sub>_ which is pro-Government and with interests to Sports news, and user _I<sub>2</sub>_ which is also pro-Government and interested on Entertainment news. Our earlier checks tell us that Sports
and Politics news usually get a good spot in the ranking. That’s a hint that Google News thinks these topics matter on a larger scale. Now, following our plan, **we’d expect a topic distribution vector with high news frequency at the top (for pro-Government, let’s say position 1) and another spike further down the line (let’s call that Sport at position 4)**. Same way for user 2.

**We calculate _User Viewpoint Similarity Index_** between two users _I<sub>1</sub>_ and _I<sub>2</sub>_ , **as the _weighted cosine similarity_ between Topic Distribution  _I<sub>1</sub>_ and Topic Distribution  _I<sub>2</sub>_**. We repeat this for all pair of users to construct the complete matrix.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/596ea260-26b4-492e-9aa1-fcf5fad17318" style="width: 70%; margin-right: 10px;" />
    </div>
    <p><em>Figure 4: User Viewpoint Similarity Index Matrix.</em></p>
</div>

Our observations indicate that _User Viewpoint Similarity Index_ is higher if the users have same macro news choices, such as, _I<sub>1</sub>_ and _I<sub>2</sub>_ have a similarity score of 0.43. We can also see that user _I<sub>3</sub>_ and _I<sub>4</sub>_ have a similarity score of 0.38, whereas it varies significantly if the users like political news but with different stance. Additionally, we observe that Entertainment is a very popular news topic worldwide and has huge number of recommended news which affects User Viewpoint Similarity Index.

#### User Viewpoint Representativeness Score
We use _**User Viewpoint Representativeness Score**_ to understand **the level of specificity in news recommendation for an user, and how it varies than the news articles published that day**.

We calculate _User Viewpoint Representativeness Index_ as **the KL-divergence between Topic Distribution (_I<sub>1</sub>_) and Topic Distribution (_I<sub>H</sub>_), where Topic Distribution (_I<sub>1</sub>_) and Topic Distribution (_I<sub>H</sub>_) represents the distribution of the news articles with respect to different topics for _I<sub>1</sub>_ and news published that day (Homepage of Google News), respectively**. **The Kullback-Leibler (KL) divergence, also known as relative entropy, is a measure of how one probability distribution diverges from a second, expected probability distribution**.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/80a6754f-bc17-43c8-ac71-d2ba38e24c19" style="width: 45%; margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/6fdf4e59-2d06-4544-9ac7-822e1fc8ef9f" style="width: 45%;" />
    </div>
    <p><em>Figure 5: User Viewpoint Representativeness Score for USA and India, respectively.</em></p>
</div>

**The Jensen-Shannon Divergence (JSD)** is a measure of **similarity between two probability distributions**. It is derived from the Kullback-Leibler. Our observations indicate that the most of the users have very low _User Viewpoint Representativeness Score_, i.e., very high KL divergence score around 0.4 to 0.7.

#### User Stance Index
For user political leaning based Viewpoint Analysis, we study **the fraction of news articles recommended to an user for a particular political leaning on a day**. This provides us with an understanding of **variance in the recommended news across different users based on their news preferences and political leaning**.

For example, on any particular day, we calculate the number of news articles recommended to an user with respect to a particular political leaning. Therefore, suppose, for political leaning as Republican, we calculate the number of news articles with Republican political leaning recommended to an user.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/624830df-5a98-48a0-b764-40a08c6c4ce7" style="width: 30%; margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/df737a68-66c2-43c9-96fa-60bb793dcb74" style="width: 30%;" />
        <img src="https://github.com/user-attachments/assets/551c338f-b4be-4f90-806c-4e3b9bd50e67" style="width: 30%;" />
    </div>
    <p><em>Figure 6: User Stance Index for USA.</em></p>
</div>

Our observations indicate that users are recommended news that match with their preferred political leaning irrespective of location. Additionally, we observe that Google News recommends democratic news and neutral political news rather than republican news to an user with no political leaning or news reading behavior.

### New Users in Echo Chambers
In order to understand this timeline of generic news recommendation to specifically tuned news recommendation, we perform an experiment where **we create a batch of 6 new users (_N<sub>1</sub>_ to _N<sub>6</sub>_), of different news topic choices, and simulate how their news recommendation behavior changes daily**. 

We consider the macro news topic of _N<sub>1</sub>_ as Entertainment, _N<sub>2</sub>_ as Sports, _N<sub>3</sub>_ as Technology, _N<sub>4</sub>_ as pro-Government leaning, _N<sub>5</sub>_ as pro-Opposition leaning and _N<sub>6</sub>_ has Neutral political leaning.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/e83b6a83-b7ed-45e8-add7-47528cea3e50" style="width: 45%; margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/91152cc7-328c-4012-856f-8f11c73f683c" style="width: 45%;" />
    </div>
    <p><em>Figure 7: New Users’ News Temporal Evolution.</em></p>
</div>

Our observations indicate that it takes more number of days for an user with specific political leaning to get a majority of its news feed aligned to that specific political leaning than an user who reads Entertainment or Sports based news. This is logically attributed to the need for a sufficiently large number of news articles. The algorithm must discern not only the user’s interest in political topics (which is relatively straightforward to differentiate from other topics) but also determine the specific political party alignment of the user. In contrast, users interested in Sports or Entertainment-related news find their preferences recognized more swiftly, as these topics are more easily distinguishable.

### Case Study: Susceptibility to Propaganda News
In this section, we delve into examining whether there exists a correlation between the user’s chosen news topics and their susceptibility to being recommended news with propaganda. 

Propaganda, in the context of news articles, refers to the dissemination of biased or misleading information intended to shape public opinion or promote a particular agenda. It often involves the use of persuasive techniques to influence individuals’ beliefs or attitudes, rather than presenting objective and unbiased reporting. Our analysis aims to uncover whether users who engage with specific news topics, especially those related to politics, might encounter a higher likelihood of receiving news articles with propagandistic elements. 

We follow Morio et al. [[Morio et al., 2020]](https://aclanthology.org/2020.semeval-1.228/) to detect whether a news article has propaganda or not. Morio et al. [[Morio et al., 2020]](https://aclanthology.org/2020.semeval-1.228/) proposes **a transformer based model which utilizes the sentence embedding coupled with named entity and pos embedding to detect whether a sentence has propaganda or not and consider a news article has propaganda if any of the sentences is propagandastic in nature**.

Let's take an example: "Also, as this is an unprecedented moment in papal history, perhaps the unprecedented step of recalling Benedict XVI to the Chair of St. Peter should also be considered sooner rather than later, before this crisis gets any more out -of- control than it already is". This provided text does have a propagandistic nature.  The text uses strong emotional language like "unprecedented moment," "crisis," and "out-of-control" to evoke a sense of urgency and fear; it suggests a specific course of action—recalling Benedict XVI to the Chair of St. Peter. Also, it describe the situation as a "crisis" and "out-of-control" can be seen as an exaggeration, which is a common technique in propaganda to stress the importance of the issue and the proposed solution.

In the subsequent section on text preprocessing, we will detail the methodology employed to extract text from news articles sourced from Google News. For the present discussion, it is sufficient to acknowledge that we have obtained the raw text of these articles, which serves as the basis for our propaganda analysis. Using this raw text, **we will apply a classifier designed to assess and verify, with a specified level of accuracy, whether the articles contain elements of propaganda**.

```python
data = pd.read_csv('output.csv')
```

the ```output.csv``` file is structured as follows:

```python
data.head()
```

| embeddings                                                                                         | label |
|----------------------------------------------------------------------------------------------------|-----------|
| Former Apostolic Nuncio to the United States Accuses Pope of McCarrick Cover-up, Calls on Francis to Resign | 0         |
| In this tragic moment for the Church in various parts of the world — the United States, Chile, Honduras, Australia, etc. | 1         |
| — bishops have a very grave responsibility | 1         |
| I am thinking in particular of the United States of America , where I was sent as Apostolic Nuncio by Pope Benedict XVI on October 19 , 2011 , the memorial feast of the First North American Martyrs | 0         |
| The Bishops of the United States are called , and I with them , to follow the example of these first martyrs who brought the Gospel to the lands of America , to be credible witnesses of the immeasurable love of Christ , the Way , the Truth and the Life | 0         |

The file is an example of csv containing, **in the left column called ```embeddings```, examples of text** containing, **if the element is propaganda, in the column corresponding to the same row ```label``` the value 1**. **If the text is not propaganda material, then label will contain 0**.

We use this file as a dataset to train the classification model for news articles.

```python
X, y = data['embedding'], data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

We split the dataset into training set and test set. Our goal is to predict whether future text blocks will contain propaganda, so whether future labels will be 0 or 1. **We assign X (input) to embeddings and y (output) to labels, and build a training set consisting of 80% of the original dataset, and a test set of 20% of the original dataset**.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

This code block loads a pre-trained tokenizer for the **Bert model**. The BERT tokenizer is a tool used to preprocess text for **BERT (Bidirectional Encoder Representations from Transformers)**; the BERT tokenizer is a preprocessing tool designed for **transforming raw text into a format that the BERT model can work with**. It uses advanced techniques such as **WordPiece tokenization to break text into smaller units (tokens)**, including subwords and characters, rather than just simple word splits. **Each token is mapped to an index in a predefined vocabulary built from a vast corpus, ensuring that the model understands a wide range of language patterns**. Additionally, **the tokenizer incorporates special tokens like [CLS] for classification tasks** (we will see its use shortly).

**DistilBERT** is a smaller, faster version of the BERT.

```python
final_model = RandomForestClassifier()

embedding_list = []
label_list = []

batch_size = 32
total_samples = len(X_train)
```

The classification model we use for our analysis is **the RandomForestClassifier**. The RandomForestClassifier is a machine learning algorithm used for classification tasks. It belongs to the family of **ensemble methods**, specifically based on the concept of **"bagging" (Bootstrap Aggregating)**: it builds a collection of decision trees during training and merges their outputs to make a final prediction. The idea is that combining the predictions of multiple trees improves overall performance and robustness compared to a single decision tree; each tree in the forest is trained on a random subset of the training data, which is sampled with replacement (bootstrap sampling). Each tree is also trained using a random subset of features for each split, which introduces further diversity among the trees.

We initialize the embeddings and label lists and populate them consequently. Data is processed in a 32 samples group per time.

```python
train_batch = X_train.values.tolist()[start_idx:end_idx]
tokenized_batch = tokenizer(train_batch, padding=True, truncation=True, return_tensors="pt")
```

The batch is tokenized using the pre-trained distilbert model, ensuring that the sequences are all the same length and truncating those that are too long.

```python
hidden_batch = model(**tokenized_batch)
cls_batch = hidden_batch.last_hidden_state[:, 0, :]
```

It runs the model on the tokenized batch, and extracts **the embeddings**. Embeddings are **high-dimensional numerical representations of text, which capture the semantic meaning of words or sentences**. They are processed by machine learning models for tasks such as classification, regression, etc. because clearly **AI models require numerical data input, and raw data is not compatible**. 

```hidden_batch``` is the output of the distilbert model which includes **the last layer hidden representation for all tokens in the sequence**. In particular, **the CLS token** is used as **the representation of the entire sequence** (BERT is designed to capture the aggregate meaning of the entire input sequence): the CLS token representation is a summary of the entire sequence.

There is no need to convert labels into embeddings, because they are already numeric values, and mostly binary. We repeat the same operation for the test set. 

```python
data_test = pd.read_csv('Data/test_embeddings_and_labels.csv')
data_train = pd.read_csv('Data/embeddings_and_labels.csv')

data_train['embeddings'] = data_train['embeddings'].apply(lambda x: ast.literal_eval(x))
data_test['embeddings'] = data_test['embeddings'].apply(lambda x: ast.literal_eval(x))

X = data_train['embeddings'].to_list()
y = data_train['labels'].to_list()

X_test = data_test['embeddings'].to_list()
y_test = data_test['labels'].to_list()

final_model.fit(X, y)
```

The function converts columns of embeddings from a string representation to a list of numbers: often, when embeddings are saved to a csv file, they are often converted to strings for ease of saving. In order to use them in a template, they must be in a numeric format. 

Finally, **the RandomForestClassifier model is trained with the input data and corresponding labels**.

```python
y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
```
```
Accuracy: 0.7614471452798192
F1 Score: 0.7199243090481391
```

The model correctly classified 76% of the examples in the test set, and an F1 score of 0.72 indicates that the model balances precision and recall well.

Now, we generate the embeddings for the dfs (that is, **we generate the embeddings starting from the text of news articles**).

```python
tokenized_data = tokenizer(dataset["sent"].values.tolist(), padding = True, truncation = True, return_tensors="pt")
hidden_data = model(**tokenized_data) #dim : [batch_size(nr_sentences), tokens, emb_dim]
cls_data = hidden_data.last_hidden_state[:,0,:]

y_data = final_model.predict(x_data)
dataset['labels'] = y_data
```

The texts of the articles contained in ```dataset["sent"]``` are tokenized and passed to the tokenizer of the distilbert model. the CLS layer containing a summary of the entire sequence is extracted, and the model is used to predict the labels based on the previous embeddings.

Finally, the predicted labels are saved in ```dataset["labels"]```. Let's take a look into the values for any user:

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/e8fe7d6e-2ca1-4f71-9440-1a9169887578" style="width: 45%;" />
    </div>
    <p><em>Figure 8: News Recommendation with respect to Macro Topic for Users from India.</em></p>
</div>

We observe that political news has higher likelihood to have propaganda and furthermore, our previous experiments confirm that the users are recommended news on the basis of their macro news topic, we can conclude that users with macro news topic with a specific politics leaning has higher likelihood to be recommended propaganda news. 

This study further indicates that this can be of huge concern and requires users to be aware of their news feed and susceptibility. It also highlights the requirement to develop news recommender approaches for Google News which specifically ensures prevention of propaganda based news recommendation to users and specifically, to users with a political leaning as they are more susceptible than their counterparts.

## Content Analysis and Retrieval
Until now, we explored the evolutionary trend of news, starting from an empty pool of news and gradually building a dataset containing a collection of news for each user.

In this section, we examine **the tone used within the article, analyse the use of words and adjectives, dissect individual components of sentences, and scrutinize the overall composition of sentences**. Through these analyses, **we generate different scores to determine whether an article is inclined towards or against a particular topic it addresses**.

### Extracting Urls from Dataset
In essence, we create a dataset containing news articles repository clicks along with their respective links. These links serve as the entry points to the actual news articles.

In the process of preparing the textual content for subsequent analyses, our initial step involves **the extraction and preprocessing of the article text**. By copying links from Google News, being a news aggregator, before actually entering the news site, the user first enters the aggregator site, which consequently takes the user to the original news site. And this hyperlink refers to the link that leads to the real news. 

By giving only the news aggregator link seen before as input, it's impossible for the system to automatically access the original article. First, **we need to extract the real link from the Google News repository**:

```python
# Set headers to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # Send a GET request to the URL with headers
    response = requests.get(url, headers=headers)
    response.raise_for_status()
```

With ```requests``` library, we can interact with web services, **retrieve data from URLs, and perform various HTTP operations**. With a get request, passing the Google News article as a parameter,**we are able to obtain the original URL via a get request**.

The extraction process encompassed retrieving essential components of the article, including **the title, subtitle (if present), description (if present) and the text of the article**. The description, in this context, refers to the paragraph typically positioned after the subtitle but before the main text.

```python
# Extract the title, subtitle, description, and main text
    title_element = soup.find('title')
    title = title_element.text.strip() if title_element else ""

    subtitle_element = soup.find('meta', attrs={'name': 'description'})
    subtitle = subtitle_element['content'].strip() if subtitle_element and 'content' in subtitle_element.attrs else ""

    description_element = soup.find('meta', attrs={'name': 'og:description'})
    description = description_element['content'].strip() if description_element and 'content' in description_element.attrs else ""
```

We also find the main text elements based on **the HTML structure** of the page:

 ```python
    main_text_elements = soup.find_all('p')
    main_text = "\n\n".join([element.text.strip() for element in main_text_elements if element.text.strip()])

    # Set the subtitle to the description if it is empty
    if not subtitle:
        subtitle = description.strip()

    # Concatenate the extracted strings
    article_text = f"{title}\n\n{subtitle}\n\n{main_text}"
```

We combine the final text as a unique string to be analysed.

### Text Pre-processing Overview
It involves a series of analyses aimed **at refining raw text data, transforming it from unstructured text strings into analyzable objects**.

This process begins with cleaning the text, which entails removing irrelevant elements such as punctuation, special characters, and stopwords. After cleaning, the text undergoes **tokenization, breaking it down into individual words or tokens**:

```python
# Tokenize the text into sentences
sentences = sent_tokenize(article)

# Print out each sentence
for sentence in sentences:
    print(sentence)
```
```
Republicans respond after IRS whistleblower says Hunter Biden investigation is being mishandled

Members of Congress are calling for more transparency from the Biden administration after an IRS whistleblower said an investigation into Hunter Biden is being mishandled.
```

Once we split the different sentences, and within the sentences obtained the individual tokens corresponding to each word and punctuation mark, we continue by applying **a stop word removal operation**.

It consists in a crucial step that involves the elimination of common words, known as stop words, from a given text.

```python
for i, sentence in enumerate(sentences):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    
    # Identify the stop words in the sentence
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stop_words_found = [word for word in words if word.lower() in stop_words]
```
```
Sentence  1
Total words: 38
Filtered words: ['Republicans', 'respond', 'IRS', 'whistleblower', 'says', 'Hunter', 'Biden', 'investigation', 'mishandled', 'Members', 'Congress', 'calling', 'transparency', 'Biden', 'administration', 'IRS', 'whistleblower', 'said', 'investigation', 'Hunter', 'Biden', 'mishandled', '.']
Number of filtered words: 23
Stop words identified: ['after', 'is', 'being', 'of', 'are', 'for', 'more', 'from', 'the', 'after', 'an', 'an', 'into', 'is', 'being']
Number of stop words identified: 15
```

#### Stemming Analysis
**Stemming** helps in standardizing words and reducing them to a common root, making it easier to analyse and process text data. **It involves removing prefixes or suffixes from words to obtain the root form**, even if the resulting stem may not be a valid word. 

```python
# Create a Porter stemmer object
stemmer = PorterStemmer()

words = word_tokenize(article)

# Perform stemming on each word using the Porter stemmer
stemmed_words = [stemmer.stem(word) for word in words]

# Combine the stemmed words back into a single string
output_text = ' '.join(stemmed_words)

# Write the output text to a new file
# with open('output.txt', 'w') as f:
#    f.write(output_text)

print(output_text)
```
```
republican respond after ir whistleblow say hunter biden investig is be mishandl member of congress are call for more transpar from the biden administr after an ir whistleblow said an investig into hunter biden is be mishandl .
```

#### Lemmatization
Unlike stemming, **lemmatization** considers **the context of the word and aims to transform it into a valid word lemma**.

```python
# Tokenize the input string
tokens = nltk.word_tokenize(output_text)

# Define the stop words to be removed
stop_words = set(stopwords.words('english'))

# Remove stop words
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(filtered_tokens)
```
```
['republican', 'respond', 'ir', 'whistleblow', 'say', 'hunter', 'biden', 'investig', 'mishandl', 'member', 'congress', 'call', 'transpar', 'biden', 'administr', 'ir', 'whistleblow', 'said', 'investig', 'hunter', 'biden', 'mishandl', '.', 'lawmak', 'capitol', 'hill', 'call', 'biden', 'administr', 'held', 'account', '``', 'block', '``', 'congress', 'public', 'learn', 'biden', 'famili', 'member', '’', 'busi', 'deal', 'china', '.', 'congression', 'outcri', 'come', 'whistleblow', 'within', 'intern', 'revenu', 'servic', 'alleg', 'investig', 'hunter', 'biden', 'mishandl', 'biden', 'administr', '.'
```

### Text Analytics Techniques
In this section, we leverage NLP text preprocessing techniques to comprehensively analyse the articles presented to the user. **Text Analytics**, a subset of NLP studies, **involves the automated extraction of valuable insights and patterns from unstructured text data**. Its primary objective is to transform textual information into a structured format that can be subjected to various analyses. 

#### Sentiment Analysis
The primary goal of **Sentiment Analysis** is **to categorize the expressed opinions in the text as positive, negative, or neutral, providing valuable insights into the subjective viewpoints of individuals**. 

In the context of social media, Sentiment Analysis **can be employed to assess how users feel about a particular brand, product launch, or social issue**. By evaluating the sentiments expressed in customer reviews, companies can gain actionable insights to improve their products, services, or overall customer experience.

```python
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_no_ssplit=False, max_split_size_mb=16)
```

For our analysis, we exploit four different sentiment metrics: **the Stanza library, Vader, MPQA and SentiWordNet**. Below, we show the usage of Stanza as an example. For the full usage, see the code.

```python
def get_sentiment_scores(text, nlp):
    doc = nlp(text)
    sentiment_scores = []
    for sentence in doc.sentences:
        sentiment_scores.append(sentence.sentiment)
    if len(sentiment_scores) == 0:
        return None
    else:
        return {
            'average': sum(sentiment_scores) / len(sentiment_scores),
            'maximum': max(sentiment_scores),
            'sd': statistics.stdev(sentiment_scores),
            'minimum': min(sentiment_scores)
        }

sentiment_scores = get_sentiment_scores(article, nlp)
print(sentiment_scores)
```
```
{'average': 0.7804878048780488, 'maximum': 1, 'sd': 0.4190581774617469, 'minimum': 0}
```

The pipeline consists of models ranging from tokenizing raw text to performing syntactic analysis on the entire sentence. The design is devised keeping the diversity of human languages in mind by data-driven models that learn the differences between languages. Besides, the components of Stanza are highly modular and reuses basic model architectures, when possible, for compactness.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/5438b812-0e97-4da5-8b28-687f6f3da6f8" style="width: 45%;"margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/32a65656-6293-4227-9f75-ccea352c4501" style="width: 45%;" />
    </div>
    <p><em>Figure 9: Sentiment Score for Users from India</em></p>
</div>

Our observations indicate that a news article has generally more than number of positive sentences followed by neutral sentences with very few negative sentences. Therefore, given _k_ number of news articles recommended to an user, there is a high probability that the sentiment score of the majority of _k_  news articles is higher than 0.5 due to the presence of positive sentences followed by few news articles which have score less than 0.5.

#### Readability Analysis
**Readability Analysis** is the evaluation of how easily a piece of text can be understood by readers. It involves assessing various linguistic and structural features of the text to determine its readability level. For our analysis, we exploit 8 different readability metrics, **including Flesch Kincaid Grade Level** and **Gunning Fog**, but for our final analysis we considered keeping only 5 for our final considerations.

```python
f = r.flesch()

print("Flesch Reading Ease Score |     Classification     ")
print("-------------------------|------------------------")
print("         0-29             | Very Difficult          ")
print("         30-49            | Difficult               ")
print("         50-59            | Fairly Difficult        ")
print("         60-69            | Standard                ")
print("         70-79            | Fairly Easy             ")
print("         80-89            | Easy                    ")
print("         90-100           | Very Easy               ")

print()
print()
# Print the readability score, ease value, and estimated reading levels
print("The Flesch Reading Ease score of the article is:", f.score)
print("The article is classified as:", f.ease)
```
```
Flesch Reading Ease Score |     Classification     
-------------------------|------------------------
         0-29             | Very Difficult          
         30-49            | Difficult               
         50-59            | Fairly Difficult        
         60-69            | Standard                
         70-79            | Fairly Easy             
         80-89            | Easy                    
         90-100           | Very Easy               


The Flesch Reading Ease score of the article is: 33.18228541964146
The article is classified as: difficult
```

Therefore, we conclude that the readability scores of an user has no pattern among users irrespective of their macro, micro news topic, location and the readability metric.

#### Part-Of-Speech (POS) Tagging
**The Part-Of-Speech (POS) Tagging** is a process in NLP that involves labeling each word in a text corpus with its corresponding part of speech, such as noun, verb, adjective, etc.

```python
sentences = sent_tokenize(article)

total_adjectives = 0
total_words = 0

for sent in sentences:
    words = word_tokenize(sent)
    tagged_words = pos_tag(words)
    num_adjectives = len([word for word, tag in tagged_words if tag.startswith('JJ')])
    total_adjectives += num_adjectives
    total_words += len(words)

avg_adjectives = total_adjectives / total_words

print(f"Total words: {total_words}")
print(f"Total adjectives: {total_adjectives}")
print(f"Average number of adjectives in the article: {avg_adjectives:.2f} in this article.")
```
```
Total words: 830
Total adjectives: 61
Average number of adjectives in the article: 0.07 in this article.
```

For this, we study **the frequency of words, frequency of stop words and frequency of adjectives**, respectively for each recommended news article.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/5857c2fe-6f28-492b-80f7-bc76e6047d25" style="width: 45%;"margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/1d4bb883-fec2-43d0-80cf-8f986836ad68" style="width: 45%;" />
    </div>
    <p><em>Figure 10: Average Dependency Tree Length</em></p>
</div>

Our observations indicate that there is no relationship between frequency of words and frequency of adjectives with the macro and micro news topic, i.e., most of the news articles are similar in length except few outliers which does not give any significant relationship or correlation. Additionally, we don’t find an user has higher likelihood to be recommended news articles with larger number of adjectives on the basis of the macro and micro news topic.

#### Topic Modeling
**Topic Modeling** is technique **to extract the hidden topics from large volumes of text**. It is a probabilistic model which contain information about the text; finding good topics depends on the quality of text processing, the choice of the topic modeling algorithm, the number of topics specified in the algorithm.

We exploit **Gensim LDA algorithm for topic extraction**: it considers each document as a collection of topics and each topic as collection of keywords. Once we provide the algorithm with number of topics all it does is to rearrange the topic distribution within documents and key word distribution within the topics to obtain good composition of topic-keyword distribution.

```python
bigrams = list(nltk.bigrams(filtered_tokens))
trigrams = list(nltk.trigrams(filtered_tokens))

# Print the results
print("Bigrams:")
print(bigrams)
print("Trigrams:")
print(trigrams)
```
```
Bigrams:
[('Republicans', 'respond'), ('respond', 'IRS'), ('IRS', 'whistleblower'), ('whistleblower', 'says'), ('says', 'Hunter'), ('Hunter', 'Biden') [...]

Trigrams:
[('Republicans', 'respond', 'IRS'), ('respond', 'IRS', 'whistleblower'), ('IRS', 'whistleblower', 'says'), ('whistleblower', 'says', 'Hunter'), ('says', 'Hunter', 'Biden') [...]
```

A **bigram model** is a language model that uses a history of one preceding word to predict the next word. It is a type of n-gram model, where n is the number of words in the history. A **trigram model**, on the other hand, uses a history of two preceding words to predict the next word.

By incorporating more context into the model, they are able to better capture **the meaning of the text** and make more accurate predictions.

```python
# Create Dictionary 
id2word = corpora.Dictionary(lemmatized_bigrams)  

# Create Corpus 
texts = lemmatized_bigrams

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus)
```
```
[[(0, 1), (1, 1)], [(1, 1), (2, 1)], [(2, 1), (3, 1)], [(3, 1), (4, 1)], [(4, 1), (5, 1)], [(5, 1), (6, 1)], [(6, 1), (7, 1)], [(7, 1)
```

This code prepares the data for topic modeling with LDA by creating a dictionary of all unique words in the corpus and a corpus object with bag-of-words representations of the documents. 

Each tuple in the output represents a bigram that has been transformed into a two-element tuple. The first element of each tuple is the ID of the corresponding bigram in the dictionary (id2word) and the second element is the count of how many times that bigram appears in the input text. For example, the first tuple (0, 1) represents the bigram ('Republicans', 'respond'), where Republicans has ID 0 in the dictionary, and respond has ID 1. The number 1 indicates that this bigram appears once in your text corpus.

```python
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
```

This code creates and trains an LDA model using the Gensim library to identify 20 topics from a given corpus of text documents. The model is configured with specific parameters to control aspects like randomness, update frequency, chunk size, number of training passes, and the alpha parameter, which influences the distribution of topics per document. The ```per_word_topics``` parameter allows the model to return information about word distributions within topics.

```python
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```
```
[(0,
  '0.004*"Biden" + 0.004*"mishandled" + 0.004*"Department" + 0.004*"either" + '
  '0.004*"choice" + 0.004*"single" + 0.004*"every" + 0.004*"need" + '
  '0.004*"potential" + 0.004*"behavior"'),
 (1,
  '0.239*"whistleblower" + 0.130*"learning" + 0.038*"held" + 0.020*"come" + '
  '0.020*"Hill" + 0.002*"whether" + 0.002*"privy" + 0.002*"protected" + '
  '0.002*"disclosure" + 0.002*"scheme"'), [...]
```

Each topic is combination of keywords and each keyword contributes a certain weightage to the topic. Keywords for each topic and weightage of each keyword using ```lda_model.print_topics()```.

We now calculate **Model Perplexity and Coherence**: Coherence is a measure of **how coherent the topics are**. Higher coherence scores indicate more coherent topics. CoherenceModel is a class in Gensim that computes the coherence of a topic model.

At the other hand, Perplexity is a measure of **how well the LDA model predicts the corpus**. Lower perplexity scores indicate better predictions.

```python
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
# a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts = lemmatized_bigrams, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```
```
Perplexity:  -21.178629464587097

Coherence Score:  0.739709594537693
```

```python
# Visualize the topics
pyLDAvis.enable_notebook(local=True)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# pyLDAvis.display(vis)
pyLDAvis.save_html(vis, 'lda_plot.html')
```

The visualization shows the topics generated by the LDA model as circles, where the size of the circle represents the prevalence of the topic in the corpus. Each topic is represented by a list of words associated with that topic, and the strength of the association is represented by the distance between the words and the center of the circle.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/62548d23-b5fa-4f23-b5d4-a0c7abcc25c0" style="width: 45%;"margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/c7a9e2d9-2946-43be-9456-1bb97eb29d04" style="width: 45%;" />
    </div>
    <p><em>Figure 11: Number of Topics in a News Article</em></p>
</div>

Our observations indicate that there is low variance in the number of topics across users. We observe that the number of topics ranges between 1 to 4 on an average irrespective of the macro and micro news topic.

#### Dependency Tree Height
Calculating **the dependency tree length** involves analyzing the syntactic structure of each sentence to determine how the words are related to each other. Dependency parsing is the process of analyzing the grammatical structure of a sentence and determining the relationships between words. Dependency tree length refers to the number of edges in the dependency tree that represents the sentence. The length of the dependency tree is an indication of the complexity of the sentence.

By finding the average, maximum, and minimum dependency tree length of a set of sentences, we can gain insights into the complexity of the text.

```python
doc = nlp(article)
depths = {}

def walk_tree(node, depth):
    depths[node.orth_] = depth
    if node.n_lefts + node.n_rights > 0:
        return [walk_tree(child, depth + 1) for child in node.children]


[walk_tree(sent.root, 0) for sent in doc.sents]
print(depths)
print(max(depths.values()))
```
```
{'respond': 0, 'Republicans': 1, 'says': 1, 'after': 5, 'whistleblower': 6, 'IRS': 5, 'mishandled': 2, 'investigation': 2, 'Biden': 1, 'Hunter': 2, [...]
```

The depths dictionary now contains the depth of each node in the dependency tree. The keys of the dictionary are the text of the nodes, and the values are the depths.

```python
def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max([walk_tree(child, depth + 1) for child in node.children], default=depth)
    else:
        return depth
    
def analyze_article(article):
    doc = nlp(article)
    depths = {}
    tree_lengths = {}
    for sent in doc.sents:
        root = sent.root
        depth = walk_tree(root, 0)
        depths[root.orth_] = depth
        tree_lengths[sent.text.strip()] = depth

    lengths = list(tree_lengths.values())
    avg_length = sum(lengths) / len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    max_depth = max(depths.values())
    max_depth_words = [word for word, depth in depths.items() if depth == max_depth]
    return tree_lengths, max_depth, max_depth_words, avg_length, max_length, min_length
```

The ```walk_tree()``` function recursively traverses the syntactic dependency tree of each sentence in the article and stores the depth of each node in the depths dictionary. The depth of a node is defined as the number of edges on the path from the node to the root of the dependency tree. The walk_tree function is called on the root of each sentence (sent.root) with an initial depth of 0.

The depths dictionary maps each token (i.e., word or punctuation symbol) in the article to its depth in the dependency tree. The keys of the dictionary are the orthographic forms (i.e., the string representations) of the tokens, and the values are the depths.

```
Dependency tree lengths:
Sentence 1: length 9
"Republicans respond after IRS whistleblower says Hunter Biden investigation is being mishandled
```

### Filter Bubbles
**Filter Bubbles** refer to the phenomenon wherein **individuals are increasingly exposed to information and perspectives that align with their existing beliefs, preferences, and interests, while being shielded from contradictory or diverse viewpoints**.

In the context of news media aggregators, such as online platforms and social media networks, algorithms curate and prioritize content based on user-specific data, including past behaviors, interactions, and demographic information.

For this type of analysis, we take note of how many news about a particular topic Google News returned to that particular user. To do this, **we build two matrices**: **the first _m<sub>1</sub>_ x _n<sub>1</sub>_ matrix, with _m<sub>1</sub>_ rows as many as the topics of users from the USA**, and **n<sub>1</sub> columns as many as the users from USA plus a column dedicated to the Home section** (the usefulness of the additional column will be explained in the next rows), and **a second matrix _m<sub>2</sub>_ x _n<sub>2</sub>_, with _m<sub>2</sub>_ rows as many as the topics of users from India, and _n<sub>2</sub>_ columns as many as there are users from India plus a column dedicated to the Home section**. 

**The sum of each column is equal to 10**: for each user, we take note of **how many news articles up to a maximum of 10 were presented by Google News to the users**. In this way, we can define **the numbers in the cell as the number of news articles presented to user _i_ (with _i_ ranging from U<sub>1</sub> to U<sub>m</sub>) belonging to topic _j_ (with _j_ ranging from T<sub>1</sub> to T<sub>n</sub>)**. **The first three topics of matrix 1 are, respectively, Republican Party, Democratic Party and Neutral Party**. **The first three topics of matrix 2 are, respectively, pro-Government Party, pro-Opposition Party and Neutral Party**. **The last column represents the Home column**, i.e. the number of news articles belonging to a particular topic present on the Google News Homepage.

#### Average News Document Stance
The experiment is based on these two matrices, from which we obtain two potential indices for the study on Filter Bubbles. The first is called _**Average News Document Stance**_: this index represents **the average position of the news viewed by users based on the various topics considered**. This index is calculated for each user and for each topic of interest. It’s calculated for each user as a weighted average of the scores relating to the various topics.

The first three rows of the matrix are extracted, iterating through the columns of the matrix and for each user, we create a dictionary which contains the scores relating to the various topics.  **For each topic, the relative score for the user is calculated by dividing the number of news articles related of that topic viewed by the user by the number of total news article viewed by the user across all topics**. This represents **the fraction of news relating to that topic compared to the total news viewed** by the user.

For each user, **it represents the distribution of the user’s preferences with respect to the various topics**, calculated as a weighted average of the scores relating to the individual topics.

```python
def calculate_average_news_scores(matrix):
    topics = matrix[:3]

    users = []

    for user_index in range(matrix.shape[1]):
        user = {}
        user['User'] = user_index + 1  # Per l'utente n, inizia da 1
        for topic_index, topic in enumerate(topics):
            topic_name = ''
            if topic_index == 0:
                topic_name = 'rep'
            elif topic_index == 1:
                topic_name = 'dem'
            else:
                topic_name = 'neu'
            
            user[f'{topic_name} score'] = topic[user_index] / sum(topic)
        
        users.append(user)

    return users
```
```
U1:
rep score: 0.25
dem score: 0.18
neu score: 0.00
U2:
rep score: 0.50
dem score: 0.09
neu score: 0.00
[...]
```

#### Entropy User Score
The second calculated index is called _**Entropy User Score**_. The calculation of user entropy evaluate **the diversity of that particular user’s preferences with respect to the various topics considered** in the context to of the news aggregator.

To calculate this index, we scroll through the columns of the matrix (representing the users), and for each user **we calculate the fractions relating to the number of news articles associated with each topic compared to the total news viewed by the user**, and **the entropy variations are calculated for each topic, using the Shannon Entropy formula**:

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/68077343-3f68-49f7-8de9-6b9ddad75551" style="width: 30%;" />
    </div>
</div>

where **_P(x<sub>i</sub>)_ represents the probability that the user sees news related to a given topic**. If a user does not view any news on a particular topic, the relative entropy change is considered as 0. 

Finally, the entropy changes relating to all topics are added to obtain the total entropy of the user. User entropy provides a measure of the diversity or variability of their preferences with respect to the topics considered.

```python
def calculate_user_entropy(matrix):
    users = []

    for user_index in range(matrix.shape[1]):
        user = {}
        user['User'] = user_index + 1  

        rep = matrix[0, user_index]
        dem = matrix[1, user_index]
        neu = matrix[2, user_index]

        total = rep + dem + neu
        if total == 0:
            entropy = 0  # avoid division by zero
        else:
            rep_frac = rep / total
            dem_frac = dem / total
            neu_frac = neu / total

            if(rep_frac != 0):
                rep_var = -rep_frac * np.log(rep_frac)
            else:
                rep_var = 0
            if(dem_frac != 0): 
                dem_var = - dem_frac * np.log(dem_frac)
            else:
                dem_var = 0
            if(neu_frac != 0):  
                neu_var = - neu_frac * np.log(neu_frac)
            else:
                neu_var = 0
                
            if(not(math.isnan(rep_var)) and not(math.isnan(dem_var)) and not(math.isnan(neu_var))):
                entropy = rep_var + dem_var + neu_var

        user['rep'] = rep_var
        user['dem'] = dem_var
        user['neu'] = neu_var
        users.append(user)

    return users
```
```
Rep for User U1: 0.37
Dem for User U1: 0.27
Neu for User U1: 0.00
==========================================
Rep for User U2: 0.27
Dem for User U2: 0.37
Neu for User U2: 0.00
[...]
```

Together, these two indexes can provide useful information to identify the presence of Filter Bubbles in users of a news aggregator; the first one provides an overview of the average user preferences with respect to the various topics or themes. High uniformity in preference scores suggests that users are primarily exposed to content that confirms their preexisting opinions. The second index evaluates the diversity or variability of user preferences with respect to the various topics.

<div align="center">
    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/user-attachments/assets/01a5f1d7-35e6-4a3a-a12d-6bb12c318240" style="width: 30%; margin-right: 10px;" />
        <img src="https://github.com/user-attachments/assets/d29ab296-51ee-44cf-be36-757ae9296149" style="width: 30%;" />
        <img src="https://github.com/user-attachments/assets/0eca7dd9-b236-47c2-8f97-790ddb41f701" style="width: 30%;" />
    </div>
    <p><em>Figure 12: Entropy Score for Users from USA.</em></p>
</div>

Furthermore, it seems that news about the Democratic Party is not only present in users who view favorable news about the Democratic Party, but in all users who follow political news, plus the Homepage. The news aggregator may seek to provide a broader range of political news that reflects diverse perspectives, including Democratic Party views, regardless of users’ political affiliation.

Users display varying preferences in news consumption. Those with higher entropy values (>0.37) engage with diverse Republican viewpoints, while an entropy of 0 indicates a focus on a single perspective. Democratic news preferences are relatively uniform, with entropy around 0.35, showing some diversity but less than Republican news. Neutral news preferences vary widely, with some users exploring diverse topics and others focusing on specific areas.

## Automated Collecting Data System
Collecting news for our study prove to be a time-intensive process, particularly when simulating the daily routines of multiple users engaging with diverse news articles. This task becomes even more pronounced when we need to repeat the processfor approximately 10 users for each component of the project, each requiring distinct news content for their daily interactions. For more information on how our dataset was built from scratch, please refer to the thesis or paper.

To prepare for future versions of our research and analyses, we set ourselves up strategically to avoid manually reconstructing the dataset. **We start creating an automated system for collecting news**: this system streamlines the process of gathering data, saving us from the tedious job of compiling it manually. **We identify Selenium as a powerful open-source framework for automating web browsers**, including automate interaction with page objects. 

This system facilitate the simulation of user interactions with news articles, automating tasks such as clicking buttons, navigating pages, and extracting data.

```python
# Create a new instance of the Firefox driver
custom_options = Options()
custom_options.set_preference('intl.accept_languages', 'et,en-US')  # Translating from Estonian to English language

driver = webdriver.Firefox(options=custom_options)

# navigate to the website
driver.get('https://answers.yahoo.com/')

# Wait for the consent popup to appear and accept it
try:
    popup = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//form[@class='consent-form']")))
    driver.execute_script("arguments[0].querySelector('button[name=agree]').click();", popup)
except:
    print("No consent popup found")

# Wait for the login button to be present
login_button = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID,
     'ybarAccountProfile'))
)

# Click the login button
login_button.click()

# Identifying username in the form
username = driver.find_element(By.ID, "login-username")

# Fill the username field
username.send_keys('mohamedcebrailhegedus@yahoo.com')

# Disabling the "Stay signed in" button
stay_signed = driver.find_element(By.ID, "persistent")

if stay_signed.is_enabled():
    # If the checkbox is enabled, disable it using JavaScript
    driver.execute_script('arguments[0].disabled = true;', stay_signed)

# Clicking on submit button, after inserting username
signin_button = driver.find_element(By.NAME, 'signin')
#signin_button.click()
```

Once logged in, our automated system seamlessly navigates to the Yahoo homepage, where news content is presented to the user.

The main part of our automated system is figuring out what kind of news a user really likes. By assigning specific attributes to the system, we enable the identification of objects within the web page, such as news links, that correspond to a designated string. However, **the system, operating solely on the provided string, lacks the autonomy to independently distinguish between articles supporting and opposing the user’s preferences**. This is where the integration of **Natural Language Processing analyses becomes essential**: **the system, by simultaneously executing these analyses, can gauge whether the article surpasses a predefined goodness score threshold set by us**. This threshold serves as a criteria to determine whether the article is pro or against, for example, to a particular politic party. **Subsequently, the system can make informed decisions about clicking and simulating the reading of the article based on its alignment with the user’s interests**. Once the decision is made on whether to click on a particular article, maintaining a record of the clicked news becomes a straightforward process.

Please note: the automated system is under development. The above code is provided as an example to show the reader how the dataset construction system can be automated for each user. We opted for Yahoo news aggregator due to the availability of the APIs provided by the media aggregator.

## Conclusions
The research being conducted is anticipated to contribute significantly to the advancement of the field of natural language processing (NLP) and text analytics. By exploring new methodologies and approaches, this work aims to enhance the accuracy and effectiveness of NLP models in the News Aggregators field, leading to improved capabilities in analyzing and understanding textual data about news articles. The outcomes of this research are expected to provide valuable insights and tools that will advance both academic research and practical applications in the domain of text analytics.

As a future direction, we intend to extend this work to other/his news media aggregators and include users from other/his countries. Although characterization and visualization of echo chambers is of high significance, it highlights a major flaw in news recommendation, i.e., requirement of development fair news recommender approaches for Google News Recommender such that formation of echo chambers is prevented.


<sub>Roshni Chakraborty, Ananya Bajaj, Ananya Purkait, Pier Luigi Trespidi, Tarika Gupta, Flavio Bertini, and Rajesh Sharma. 2023. Echo Chambers in News Media Aggregators. ACM Trans. Web 1, 1, Article 1 (January 2023)</sub>

<sub>Authors’ addresses: Roshni Chakraborty, University of Tartu, Tartu, Estonia, roshni.chakraborty@ut.ee; Ananya Bajaj*, Indian Institute of Technology Goa, Goa, India, ananya.bajaj.20033@iitgoa.ac.in; Ananya Purkait*, Indian Institute of Technology Goa, Goa, India, ananya.purkait.21033@iitgoa.ac.in; Pier Luigi Trespidi*, Department of Mathematical, Physical and Computer Sciences, University of Parma, Parma, Italy, pierluigi.trespidi@studenti.unipr.it; Tarika Gupta, Indian Institute of Technology Goa, Goa, India, tarika.gupta.20042@iitgoa.ac.in; Flavio Bertini, Department of Mathematical, Physical and Computer Sciences, University of Parma, Parma, Italy , flavio.bertini@unipr.it; Rajesh Sharma, University of Tartu, Tartu, Estonia, rajesh.sharma@ut.ee.</sub>

