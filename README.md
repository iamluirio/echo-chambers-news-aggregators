# Study of Echo Chambers in News Media Aggregators

## Natural Language Processing (NLP), Social Network Analysis, Being able to perform large scale data analysis.

[Google Drive directory](https://drive.google.com/drive/folders/1RNXtjfaj7-E0-XlfuzSfTpAKPmExop3F/ "Project Directory") of the project.

- Data collection: creation of several accounts, in order to automatically study how different News Media Aggregators provide different type of news to those accounts that differ in terms of combination of factors.
- state-of-art study about bias in news media aggregations.
- Definition of metrics about bias in news media aggregations.
- Identification of bias in news media aggregations.
- Identifications and descriptions of Echo Chambers through qualitative and quantitative techniques.

### Git repository 

- Code Analysis:
  - Analysis:
    - [Leaning Temporal Analysis](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/Analysis/leaningTemporal_analysis.ipynb): Analysis carried out over 5 days, the progress of the topic over the various       days is evaluated, calculating the number of news provided by google news, for each day, for each topic       and for each user. 
      1. 4 graphs for the USA, 2 evaluate the pro-republican and 2 pro-democratic trend.
      2. 4 graphs for India politics, 2 evaluate the pro-gov and 2 the pro-opp trend.
      3. 4 graphs for Macro/Micro topics, 2 evaluate the macro and 2 the micro trend.
    - [News Rate Analysis](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/Analysis/newsRate_analysis.ipynb): Analysys carried out on different days (separately) on top 20 news for each   day, for each user, and plotted a multi-bar graph for each topic.
      1. 6 graphs: 2 about politics trend, 4 about macro and micro trend.
    - [Positional User Analysis](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/Analysis/positionalUser_analysis.ipynb): Analysis carried out on 3 days, about position of a particular topic in the three different days, and compared each other.
      1. 4 graphs: 2 about to User 1 - User 6, 2 about User 7 - User 10 (respectively, for each student).
    - Text Analytics Analysis: Analysis about the position of the main topic for each user, carried out on 3 days, and about the number of tabs for the top 10 articles in the related articles for each (political) user.
    - New Progression Analysis (new users): carried out over 8 days, it containing the top 10 urls in the "For you" section page given by Google News based by interests, and the clicked links to understand the path of the evolution.
  - News Media Aggregators API:
    - [News Media Aggregators](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/News%20Media%20Aggregators%20API/newsMediaAggregators.ipynb): NewsAPI to retrieve the top headlines from various news sources based on a user's email address.
    - [Yahoo News Aggregators APIs](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/News%20Media%20Aggregators%20API/yahooNewsAggregator.ipynb): APIs for accessing their news content, and to have documentation on how to use OAuth2 for authentication and authorization on Yahoo APIs.
  - NLP:
    - Single modules:
      - [Pre-process Text](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/NLP/Single%20modules/preProcess_text.ipynb): Script to convert an article into a string, retrieving, from an url, the title, the subtitle, the description (if it exists), and the main text on af article (avoiding the ads on the side of the article, or at the end).
        1. Tokenization of sentences.
        2. Tokenization of words.
        3. Stop word removal.
        4. Pre process a text.
        5. Stemming Analysis.
      - [Sentiment Analysis](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/NLP/Single%20modules/sentiment_analysis.ipynb): 
        1. Sentiment Analysis using the Stanza Library.
        2. Sentiment Analysis using Vader Library.
        3. Subjectivity analysis using MPQA.
        4. Sentiment analysis using SentiWordNet.
      - subjclueslen1-HLTEMNLP05.tff: required for implementing the MPQA analysis. It contains the MPQA lexicon and preprocess; it can be easily looked up the polarity score of each word in the lexicon. 
      - [Gensim-LDA](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/NLP/Single%20modules/Gensim-LDA.ipynb): Technique to extract the hidden topics from large volumes of text.
        1. Removing emails, newline characters and punctuations.
        2. Tokenize words and cleanup the text, removing stop words.
        3. Bigram and Trigram models.
        4. Lemmatization.
        5. Create Dictionary and Corpus needed for Topic Modeling.
        6. Compute model Perplexity and Coherence score.
        7. LDA model.
      - d3.v5.min.js, ldavis.v1.0.0.css, ldavis.v3.0.0.js are files generated by the Gensim libraries.
      - lda_plot.html is the Gensim LDA graph plot. 
      - [Readability Analysis](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/NLP/Single%20modules/readability_analysis.ipynb): A writer might use the metrics to objectively assess the complexity of his work to determine whether it’s written at a level appropriate for his intended audience. An educational software firm might use readability metrics to recommend level-appropriate content for its students.
        1. Flesch Kincaid Grade Level.
        2. Flesch Reading Ease.
        3. Dale Chall Readability.
        4. Automated Readability Index (ARI).
        5. Coleman Liau Index.
        6. Gunning Fog.
        7. SMOG.
        8. SPACHE.
        9. Linsear Write.
      - [Part-of-speech (POS) tagging](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/NLP/Single%20modules/posTagging_analysis.ipynb): Process in natural language processing (NLP) that involves labeling each word in a text corpus with its corresponding part of speech, such as noun, verb, adjective, etc. The NLTK (Natural Language Toolkit) is a popular Python library for NLP tasks, including POS tagging.
        1. Tagging Sentences and identifying adjectives.
        2. Total words, total adjectives and average number of adjectives in the article calculations.
      - [Dependency Tree Height](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/NLP/Single%20modules/dependencies_tree_height.ipynb): calculating the dependency tree length of each sentence involves analyzing the syntactic structure of each sentence to determine how the words are related to each other. Dependency parsing is the process of analyzing the grammatical structure of a sentence and determining the relationships between words. By finding the average, maximum, and minimum dependency tree length of a set of sentences, we can gain insights into the complexity of the text.
        1. Depth of each node calculation.
        2. Building the tree.
        3. Printing the tree.
    - [Automated pre-Process Article](https://github.com/ptrespidi/echo_chambers_intership/blob/main/Code%20analysis/NLP/automated_preProcessArticle.ipynb): it contains an automated system, that performs all the tasks and the analysis in the NLP folder, after providing a list of urls about text articles. It produces a list of .txt files, one for each article, which contain the scores about the analysis. 
- Plotted graph analysis
  - Leaning Temporal Analysis: it contains the graph plots about the Leaning Temporal Analysis.
  - News Rate Analysis: it contains the graph plots about the News Rate Analysis.
  - Positional User Analysis: it contains the graph plots about the Positional User Analysis.

  

