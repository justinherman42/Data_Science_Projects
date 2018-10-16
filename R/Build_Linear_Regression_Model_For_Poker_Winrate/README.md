# Research question
+ Develop a model using linear regression to predict win rates in poker 
+ I play online poker professionally and what led me into the topic actually has to do with my Data Exploration class final project. Here is the link to that project. [Build Custom Poker Statistics Software](https://rpubs.com/justin_herman_42/385739).  
    + The short explanation for that project is; that I had to access my poker Postgres database, create a customized string of poker statistics, and automate the insertion of that string into my poker sites note file.  
+ After querying my local Postgres Database and creating custom stats, I attempt to fit models to the data.
# Data 
+ Every hand I play in poker is tracked in text format.  It is then converted behind the scenes by software into statistics describing player actions. How often someone takes an action as well as the opportunity to take that action are recorded, which allows me to create percentages represented as poker statistics.  These stats are stored inside a Postgres database.  For this project I am accessing that db.
+ The cases in this study are the players in the database
+ The dependent variable is BB/100(win rate)
+ The independent variables are quantitative (vpip, pfr, wwsf, threebet) and qualitative (vpip-pfr split into a qualitative grouping of wide and narrow gap).  See "explanation of stats" section below, taken from data 607 project, to understand what these variables represent 
+ This is an observational study.  The purpose of the project was to create a linear model to classify win rates.    
+ The population of interest is online poker players.  My data comes from multiple sites I have played on in past couple years, therefore it is the global online poker player population.  
+ Generalizability is difficult.  
  + Most of my stats come from tables I play at.  Tables aren't chosen at random.  I use careful table selection to select tables where worse players play. This likely biases the player pool.
  + Playing style and strategy at different stakes can lead to different results.  It's complicated, but poker is about capitalizing on mistakes. Different types of mistakes are likely made at different levels.  For instance, when playing a free hand of poker people play much differently than they would if they had to invest substantial money. 
  + Perhaps if the population is narrowly defined as low to mid stakes online No Limit Holdem, some of the predictions can be generalized to the population. 
+ This is an observational study; therefore the data cannot be used to prove causality.  
# Explanation of Stats
+ As this is the only poker technical area in this project, I provide a brief explanation of some poker stats. 
+ In Texas Holdem players are all given two cards and are presented with a betting decision based on only their individual cards. From there they are presented with decisions on what to do as 5 community cards come out over three more rounds of betting.   
+ There are thousands of combinations of hands and hundreds of stats to choose from, but the stats I chose are the following: 
    + VPIP = How often someone calls their hand `Or` raises/ total hands played
        + Ideal range for this stat is from (22-28)
    + PFR = how often someone raises their hand / /total hands played 
        + Ideal range for this stat is from (16-22)
    + VPIP includes the entire set of PFR 
    + VPIP_PFR = VPIP-PFR
    + WWSF = Percent of the time someone wins hand after seeing a flop
    + Threebet = After someone has already raised, the percent of the time you re-raise
    + BB/100 = how many bets a player wins per 100 hands(how much someone wins)
        + Typically any win rate above 4/bb 100 is considered a solid winning player
        + This stat can be both positive and negative, negative represents losing players

# Overview

+ Query my poker Postgres database and create customized player statistics
+ Explore these statistics and test to see if assumptions for inference are met
+ Use these statistics to run a multiple linear regression model to try and predict a player’s winrate
