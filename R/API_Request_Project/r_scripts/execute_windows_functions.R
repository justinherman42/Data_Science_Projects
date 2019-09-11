library(devtools)

## Grab windows functions
source_url("https://raw.githubusercontent.com/justinherman42/Company_project/master/r_scripts/build_database.R")
source_url("https://raw.githubusercontent.com/justinherman42/Company_project/master/r_scripts/Windows_functions.R")

### Execute yty_Growth with Revenue 
yty_growth("Revenue","financials")

### Execute yty_Growth with operating expenses
yty_growth("OperatingExpenses","financials")

## Industry level ranking
yty_ranking("Revenue","financials")
print("passed first")
## Company level ranking
yty_ranking("Revenue","financials",partition=1)

## 3 year moving average 
yearly_moving_avg()

print("")
paste( "All scripts have been executed. All csv files for the windows functions should be saved on your cwd.",getwd())
