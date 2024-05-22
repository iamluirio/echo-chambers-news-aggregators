# Study of Echo Chambers in News Media Aggregators

The advent of social media and availability of news media platforms online has drastically transformed the news consumption behavior. Nowadays, the majority of the population tends to receive the daily news feeds online rather than the printed offline newspapers. These impertinent changes in news consumption demand news media sources to understand the pulse of the readers and adapt according to the trends; while personalization offers benefits, it also raises challenges, such as the risk of creating information bubbles or **echo chambers**. Existing research works define an *echo chamber* as an isolated virtual room or environment where the users’ beliefs and interests are reinforced by repeated exposure to information that aligns with user’s specific interests. 

There are several research works which focus on understanding and identifying echo chambers in different platforms, such as, blogs [Gilbert et al., 2009], forums [Edwards, 2013] and social media platforms [Zannettou et al., 2018]. However, these existing recent research works focus on identifying echo chambers on different social media platforms and there is no research work that focuses on echo chamber in news media aggregators. Existing works on social media platforms are not directly applicable for news media aggregators as there is no social relationship among users and users have only information about the news articles recommended to them. In this study we propose different measures that can quantitatively and qualitatively study characterization of echo chambers in news media aggregators across different users.

## Git repository 
- [Automated Collecting Data System](https://github.com/ptrespidi/echo_chambers_intership/tree/main/Automated%20Collecting%20Data%20System): Collecting news for our study prove to be a time-intensive process, particularly when simulating the daily routines of multiple users engaging with diverse news articles. One significant time-consuming aspect is the meticulous attention to detail in simulating the entire user experience. We started creating an automated system for collecting news; we identify Selenium as a powerful open-source framework for automating web browsers. This system facilitate the simulation of user interactions
with news articles, automating tasks such as clicking buttons, navigating pages, and extracting data. Our system exploit **Selenium** library to simulate user login, providing a username and password corresponding to one of the simulated users. This allow us to seamlessly automate the process of accessing news articles significantly streamlining the data collection process for future iterations of our research.
  - geckodriver.log: log file generated by GeckoDriver, the WebDriver implementation for Mozilla Firefox. It provides detailed logging information about the interactions between Selenium and Firefox through GeckoDriver. 

  

