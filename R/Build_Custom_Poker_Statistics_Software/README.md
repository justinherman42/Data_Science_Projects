## Project Goals

### **Problem**
+ I play online poker professionally and a key to my success is the player stats I use to identify player habbits 
+ In order to get these stats I purchased a poker program which takes text files(Hand histories) created by my poker sites, and creates and stores stats from these files inside a SQL DB. The software then provides a customized display in live time at each table(HUD)  
+ My problem is that one of the sites that I play on, no longer offers these text files and therefore my HUD no longer works 

### **Solution** 
+ I can't recreate my HUD, however, I can Automate a one time stat insertion into my poker site's XML note file.
+ The idea will be to extract a customized string from SQL and load it into an XML file 
    + Step 1 Query Postgres DB to create a dataframe of statistics
        - This dataframe will store two columns: playername, concatenated string(playerstats)
    + Step 2 Read in Poker site's existing XML notes file
        - File consists of all poker table graphics
        - Most importantly it contains a notes box I will be able to display my concatenated string in
    + Step 3 Insert the values of of query Postgres DB into my pokersite xml file 
        - Load poker site and validate that my notes have updated with my concatenated string

## Please see Rpubs file for executed script

[Rpubs_project_link](https://rpubs.com/justin_herman_42/385739)
