# How Lockdown Affected Attention Towards Video Games

## Find our data story [here](https://corentinplumet.github.io/)

## Abstract:

During the COVID-19 lockdown, people had to find other ways to keep themselves occupied, creating a rise in video game popularity. This project aims to analyze the correlation between the severity of the lockdown with the rise and fall of various video game genres. This will be done by analyzing the language used in different Wikipedia pages for each game, as well as examining the mobility data in each country to estimate the severity of lockdown.

Our project tells a fascinating story about how different types of video games faced tough challenges. We looked at the language of Wikipedia pages and connected them with data about that country's mobility change during the pandemic (for the languages that are mainly spoken in only one country). This helps us see interesting trends. Furthermore, English Wikipedia can show us what is happening worldwide as it is the most used worldwide language and can be used to estimate worldwide trends. Our story shows how strict lockdowns relate to what video game genres people tend to play more of. This is useful information about how the video game industry is changing during global disruptions and if it had a lasting effect on various genres' popularity. This can be useful information for game developers when choosing what game genre to choose, and what target audience to make it for. Furthermore we also looked at the differences between video games and board games to see how similar or different the trends were for each due to the COVID-19 pandemic.

### Research Questions:
- Is there a consistent trend for the same genres of video games during the COVID-19 pandemic?
  - Investigate whether certain video game genres experienced similar patterns of rise in popularity or decline during the pandemic.
  - Explore potential factors influencing these trends, such as changes in user preferences, gaming habits, or social dynamics.

- What is the relationship between mobility patterns and the traffic of video game Wikipedia pages?
  - Examine how variations in mobility data, reflecting the severity of lockdown measures, correspond to changes in the viewership of video game Wikipedia pages.
  - Analyze whether increased or decreased mobility aligns with heightened interest in specific video game genres, indicating a potential connection between real-world restrictions and virtual entertainment.

- To what extent can English Wikipedia pages serve as a reliable estimate of the average video game popularity?
  - Assess the generalizability of findings from English Wikipedia pages to global trends, considering cultural and linguistic differences.
  - Explore whether language-specific Wikipedia pages provide consistent insights into the popularity or decline of video game genres, or if there are notable variations that need to be considered in the analysis.

- How did the popularity change in board games compare to online games during COVID-19?
  - Is there a trend between countries that tend to have larger families with the popularity of board games in that country (Ones that have rules that can be looked up on Wikipedia).

## Additional Datasets and Methods:

To enrich our understanding, we have included additional datasets as well as using an API to collect information from the web. These supplementary data sources are instrumental in examining the interesting relationship between external factors, such as the global pandemic, and the interest in different video game genres. These include all the frequencies at which Wikipedia pages for each video game in every language was accessed, especially during the COVID-19 pandemic.

## Analysis:

For our analysis we extract the data from all the datasets and after having done some data wrangling and preprocessing. We use it to extract representative graphs in order to answer the research questions.

### Research Question 1:

To find the potential links between the mobility data and interest increase in games from different countries, we used the google mobility data and the specific Wikipedia pageviews on game topics in the time series, with the intention to make an introductory exploration on this correlation. Furthermore, we tried to explore deeper into different game genres and analyze the different game preferences from game players in different countries. We extracted and categorized games via the Wikidata API and analyze the pageview changes during the lockdown country by country. From this analysis, we realize that the negative correlations between mobility and interest in games may exist and there are different patterns on the interest shift on games in different countries during the COVID-19 period. In this way, we can gain a good support on the statistical inferences on the datasets in our following data story.

### Research Question 2:

We used the global mobility dataset and the json dataset. For the JSON dataset we extracted the video games pageviews percentage  to be sure to capture an increase in the attention for video games and not only an increase on the internet traffic. Thanks to the mobility dataset we have been able to categorize the different country according to their lockdown intensity. With the data extracted from the json file we can then perform a Pearson correlation test to know wether yes or no the 2 time series are correlated. Also, for each group, we did a measure of the average pageviews before and after the lockdown, and use a 95% confidence interval to know if the lockdown intensity of each group had an impact on the change on the average. 

### Research Question 3:

To answer this question, we used the interventions and global mobility datasets in order to compare the pageviews of all countries. With this data we calculated the average trends for all countries and did a comparison with english in order to answer the question. Thanks to this we can say that the English Wikipedia pages are in fact very useful in order to predict the worldwide trends. Furthermore to analyze more specifc games, we used a web API in order to extract more information from the web.

### Research Question 4:

For this question, we used the ‘intervention' dataset to get the dates of the turning point events. We also used an API to fetch the pageviews of different boardgames in different languages to analyze the popularity trends during COVID-19. The 'intervention' dataset was crucial in pinpointing the COVID-19 periods for various countries, allowing for a correlation between lockdowns and changes in boardgames popularity. The comparison revealed an unsual increase in board games' popularity, possibly linked to family-oriented activities during lockdowns, and the increase of quality time in family during the end of the year festivities.

## Work Splitting:

| Student | Task |
| -------- | -------- |
| Kyan Achtari | Creating the data story. Cleaning and commenting the code. Making README. Answering Questions 3.1 & 3.2 |
| Mehdi Abdallahi | Creating the data story. Cleaning and commenting the code. Answering Question 4.1 & 4.2|
| Corentin Plumet | Creating the data story. Creating the website. Answering Questions 2.1 & 2.2 |
| Zhuofu Zhou | Creating the data story. Answering Questions 1.1 & 1.2 |
| Yichen Liu | Creating the data story. Answering Questions 1.1 & 1.2 |
