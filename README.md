# Study of Echo Chambers in News Media Aggregators
<div align="left">

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
<img src="https://img.shields.io/badge/nltk-grey?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/spacy-grey?style=for-the-badge&logo=spacy&logoColor=lightblue" />
<img src="https://img.shields.io/badge/numpy-grey?style=for-the-badge&logo=numpy&logoColor=blue" />
<img src="https://img.shields.io/badge/selenium-grey?style=for-the-badge&logo=selenium&logoColor=green" />
<img height='28' src="https://github.com/user-attachments/assets/cedbd308-fd4e-4cbc-91eb-d5992999c33f" />
</div>

The advent of social media and availability of news media platforms online has drastically transformed the news consumption behavior. Nowadays, the majority of the population tends to receive the daily news feeds online rather than the printed offline newspapers. These impertinent changes in news consumption demand news media sources to understand the pulse of the readers and adapt according to the trends; while personalization offers benefits, it also raises challenges, such as the risk of creating information bubbles or **_echo chambers_**. Existing research works define an *echo chamber* as an isolated virtual room or environment where the users’ beliefs and interests are reinforced by repeated exposure to information that aligns with user’s specific interests. 

There are several research works which focus on understanding and identifying echo chambers in different platforms, such as, blogs [[Gilbert et al., 2009]](https://ieeexplore.ieee.org/document/4755503), forums [[Edwards, 2013]](https://philpapers.org/rec/EDWHDP) and social media platforms [[Zannettou et al., 2018]](https://arxiv.org/abs/1802.05287#:~:text=In%20August%202016%2C%20a%20new,suspended%20from%20other%20social%20networks.). However, these existing recent research works focus on identifying echo chambers on different social media platforms and there is no research work that focuses on echo chamber in news media aggregators. **Existing works on social media platforms are not directly applicable for news media aggregators as there is no social relationship among users and users have only information about the news articles recommended to them**. In this study, we propose different measures that can quantitatively and qualitatively study characterization of **echo chambers in news media aggregators across different users**.

## Preliminary Information
Below, we give some preliminary information about the study elements that we used for our research, so that if the user wants to replicate the tools used, he must know how to use them, and which input objects to use.
- News Media Aggregator: 

## :ledger: Index
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Morphing Process](#morphing-process)
  - [Identifying the Facial Landmarks](#identifying-the-facial-landmarks)
  - [Delaunay Triangulation](#delaunay-triangulation)
  - [Calculating Linear Interpolation for Morphed Image](#calculating-linear-interpolation-for-morphed-image)
  - [Getting Delaunay Indexes and Reshaping](#getting-delaunay-indexes-and-reshaping)
  - [Morphing Triangles and Images](#morphing-triangles-and-images)
  - [Results](#results)
 
## Usage
```
git clone https://github.com/ptrespidi/echo-chambers-news-aggregators.git
```


We show the code to be readily utilized for testing purposes, allowing you to explore the presence of echo chambers based on our analysis within a defined text-containing system. This setup also enables you to experiment with the tools we employed throughout our study; this will aid in replicating our methodology and verifying the robustness of our findings in your own investigations.

## Preliminary Information

