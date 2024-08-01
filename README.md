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

## Important Considerations
Before continuing, we recommend paying close attention to the small notes at the bottom of the page. We would like to emphasize that the chronological snapshot of the start of 2023 in politics reveals a period marked by significant events that undoubtedly influenced the dynamics of our study. It is important to recognize that every analysis involving experiments and the collection of data from temporal sources inherently carries a specific temporal timestamp. This timestamp denotes the period during which the project is conducted and the data is gathered.

The temporal aspect of our study is not merely a logistical consideration; it profoundly influences the nature and interpretation of the data collected. As a result, the data collected during this period may inherently carry a bias, reflecting the heightened emotions, polarization, and discourse surrounding the impeachment proceedings. Additionally, it is crucial to note that we do not intend to personally judge whether the news aggregator favors the creation of echo chambers. Instead, we present our experiments and analyses, allowing the reader to draw their own conclusions based on the evidence provided.

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
As Google News ranks the news articles based on relevance to the user, we explore the positional homophily in news recommendation for an user, i.e., we study the first position of any news article that belong to the user’s macro news topic.
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
We use _**User Viewpoint Representativeness Score to understand the level of specificity in news recommendation for an user, and how it varies than the news articles published that day**_.

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

Finally, the RandomForestClassifier model is trained with the input data and corresponding labels.

```python
y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
```
```
Accuracy: 0.7614471452798192
F1 Score: 0.7199243090481391
```

The model correctly classified 76% of the examples in the test set, and an F1 score of 0.71 indicates that the model balances precision and recall well.



