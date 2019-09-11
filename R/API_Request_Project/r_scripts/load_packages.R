
### Library import
library(httr)
library(jsonlite)
library(lubridate)
library(tidyverse)
library(RMariaDB)
library(DT)
library(knitr)
library(kableExtra)
library(devtools)
library(sys)

## Build df displaying package versions  
pack_ver1 <- cbind("httr",as.character(packageVersion("httr")))
pack_ver2 <- cbind("jsonlite",as.character(packageVersion("jsonlite")))
pack_ver3 <- cbind("lubridate",as.character(packageVersion("lubridate")))
pack_ver4 <- cbind("tidyverse",as.character(packageVersion("tidyverse")))
pack_ver5 <- cbind("RMariaDB",as.character(packageVersion("RMariaDB")))
pack_ver6 <- cbind("DT",as.character(packageVersion("DT")))
pack_ver7 <- cbind("knitr",as.character(packageVersion("knitr")))
pack_ver8 <- cbind("kableExtra",as.character(packageVersion("kableExtra")))

pack_versions <- as.data.frame(rbind(pack_ver1,pack_ver2,pack_ver3,pack_ver4,pack_ver5,pack_ver6,pack_ver7,pack_ver8))
colnames(pack_versions) <- c("Package Name", "Version")
print(pack_versions)

print("Packages were properly loaded")