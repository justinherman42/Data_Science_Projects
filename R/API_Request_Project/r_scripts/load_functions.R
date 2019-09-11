### Functions
source_url("https://raw.githubusercontent.com/justinherman42/Company_project/master/r_scripts/load_packages.R")

### Function accesses financial data API and returns a dataframe ready to be inserted into SQL 
### Function takes companies stock tag as an input string  

json_to_df <- function(tag)
{
    ## Build url
    url <- paste("https://financialmodelingprep.com/api/v3/financials/income-statement/",tag,"?period=quarter",sep="")
    headers = c('Upgrade-Insecure-Requests' = '1')
    params = list(`datatype` = 'json')
    
    ## Make request
    result <- GET(url = url, httr::add_headers(.headers=headers), query = params)
    result<- rawToChar(result$content)
    df <- as.data.frame(fromJSON(result)[2])
    
    ## Fix table column names- remove financials. and special char "."
    colnames(df) <- gsub("financials.|\\.","",colnames(df))
    
    ## Convert Date to datetime/ rename date as reportdate
    df$Report_Date <- as.Date(df$date, '%Y-%m-%d')
    df <-  df %>%
        select(-date)
    
    ## Build tag column to identify stock ticker
    df$Company <- tag
    
    ## Build primary key column convert financial data to numeric
    df$Id <- paste(as.character(df$Report_Date),df$Company,sep="-")
    cols.num <- colnames(df)[1:31]
    df[,cols.num] <- sapply(df[cols.num],as.numeric)
    
    ## Data check for NA-
    table(is.na(df))
    return(df)
}


### Function initalizes table creation in mysql
### takes stock tag(chracter), tablename(character)
### If table already exists, will tell you to use different name or use update function instead
build_table <- function(tag,tablename)
    {
    
    ## import df from json api call function 
    df <- json_to_df(tag)
    
    ## grab credentials from credential file
    db_credentials<-"C:\\Users\\justin\\Desktop\\xmedia.cnf"
    my_sql_db<-"xmedia"
    
    ## make connection
    my_conn<-dbConnect(RMariaDB::MariaDB(),
                       default.file=db_credentials,
                       group=my_sql_db)
    
    ## Build table from df
    tryCatch(dbWriteTable(my_conn, value = df, 
                          name = tablename,
                          overwrite =TRUE,
                          row.names = FALSE) ,error= function(e){print("table can not be overwritten and already exists. Please use update table function or change name")})
    
    ## Set primary key to Companytag+Date
    res <- dbSendQuery(my_conn, paste("ALTER TABLE",tablename,"ADD CONSTRAINT websites_pk
                                 PRIMARY KEY (`Id`(40)) ;"))
    ## Disconnect
    dbClearResult(res)
    dbDisconnect(my_conn)
}


### loops through list of stock tags and updates SQL DB 
### Will not update db if any of stock data in new queried df already exists in SQL DB
update_table <- function(tags,tablename){
    for (tag in tags){
        
        ## set error checker
        skip_to_next <- FALSE
        
        ## import df from json api call function 
        df <- json_to_df(tag)
        
        ## grab credentials from credential file
        db_credentials<-"C:\\Users\\justin\\Desktop\\xmedia.cnf"
        my_sql_db<-"xmedia"
        
        ## make connection
        my_conn<-dbConnect(RMariaDB::MariaDB(),
                           default.file=db_credentials,
                           group=my_sql_db) 
        
        # insert df into SQL.  Catches primary key conflicts(duplicate data), prints stock wasnt updated, and moves on in loop
        tryCatch(dbWriteTable(my_conn, value = df, 
                              name = tablename, 
                              overwrite= FALSE,   
                              append = TRUE,                         
                              row.names = FALSE),error= function(e){skip_to_next <<- TRUE})
        if(skip_to_next) { print(paste("Company",tag, "data already exists in DB"))
            gc()
            dbDisconnect(my_conn)
            next }
        gc()
        dbDisconnect(my_conn)
        print(paste("db was updated correctly with: ",tag ))
    }
}

print("Functions were properly loaded")
