library(devtools)


## Function to grab R files from github
source_url("https://raw.githubusercontent.com/justinherman42/Company_project/master/r_scripts/load_packages.R")
source_url("https://raw.githubusercontent.com/justinherman42/Company_project/master/r_scripts/load_functions.R")

## Use your own credentials file
db_credentials<-"C:\\Users\\justin\\Desktop\\xmedia.cnf"


### Call functions and build db {.tabset .tabset-fade}
df <- json_to_df("USB")
top_10_banks <- c('WFC', 'PNC', 'BBT', 'STI', 'KEY', 'MTB', 'HBAN', 'ZION', 'CMA', 'FITB')
## build a table in sql
build_table('USB',"financials")
update_table(top_10_banks,"financials")

### Explore SQL DB

## practice some query with db
db_credentials<-"C:\\Users\\justin\\Desktop\\xmedia.cnf"
my_sql_db<-"xmedia"
my_conn<-dbConnect(RMariaDB::MariaDB(),
                   default.file=db_credentials,
                   group=my_sql_db) 

## print out description of table
print(dbGetQuery(my_conn, "DESCRIBE financials;"))

## Check for NA values
res <- dbGetQuery(my_conn, "SELECT * FROM financials;")
table(is.na(res))

## look at sql db
res <- dbGetQuery(my_conn, "SELECT * FROM financials;")
datatable(res)
gc()
dbDisconnect(my_conn)

