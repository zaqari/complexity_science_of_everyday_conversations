######################################
#### TO ADD NEW VARIABLES
######################################
# If you are looking to add additional
#  variables to the script, you must
#    (1) Modify the script to estimate
#        the beta values for those vars
#    (2) Add the column/vec in the df
#        containing those vars in the
#        data definitions section.
#    (3) Initialize the beta estimate
#        params in myinits.
######################################

#clears workspace:
rm(list=ls())

# sets working directories:
# setwd("C:/home/zprosen/d/rtest")
file_path <- "./unit-test-analysis.csv"

library(R2jags)
df <- read.csv(file_path)

##### create new columns
df$age_dif <- df$x_age - df$y_age
df$pol_dif <- df$x_politics - df$y_politics
df$status_dif <- df$x_i_think_my_status - df$x_i_think_your_status
df$same_gender <- as.integer(df$x_sex == df$y_sex)
df$same_race <- as.integer(df$x_race == df$y_race)
df$t_delta <- df$y_turn_id - df$x_turn_id

new_cols <- c("age_dif","pol_dif", "status_dif","same_gender","same_race","t_delta")

##### data definitions (where in the dataframe we grab each value)
nx <- df$nx
ny <- df$ny
H <- df$Hxy
group <- df$x_turn_id
GROUPS <- length(unique(group))

x <- data.matrix(df[new_cols])

PI <- pi
K <- length(new_cols)
GROUPS
ROWS <- length(df$Hxy)

data <- list('nx', 'ny', 'x',
             'PI', 'K',
             'H', 'ROWS',
             'group', 'GROUPS'
)

### initializing parameter values
# A quick lesson to the wise . . . these initial values
#  need to be within the range for the values in the script.
#  Be aware of that.
myinits <- list(
  list(
       nxbeta=.01, nybeta=.01,
       mu=c(
         mean(df$age_dif),
         mean(df$pol_dif),
         mean(df$status_dif),
         mean(df$same_gender),
         mean(df$same_race),
         0
       ),
       gamma=c(rep(.001,K)),
       phi=c(rep(.001,K)),
       sigma_=c(rep(.01,K))
       # Sigma=matrix(rep(.5, K), nrow = K, ncol = K)
  )
)

parameters <- c(
  'e','Sigma', 'sigma_',
  'S', 'Sinv', 'L',
  'sinprod', 'mu', 'lambda',
  'nxbeta', 'nybeta'
)

samples <-jags(data, inits=myinits, parameters,
               model.file= "script.txt",
               n.chains=1, n.iter=3000,
               n.burnin=1000, n.thin=1, DIC=T)

# ### save outputs....
print(dim(samples$BUGSoutput$sims.list$S))
print(apply(samples$BUGSoutput$sims.list$sigma_, 2, mean))
print(log(apply(samples$BUGSoutput$sims.list$L, 2, mean)))
print(apply(samples$BUGSoutput$sims.list$mu, 2, mean))
print(apply(samples$BUGSoutput$sims.list$lambda, 2, mean))
print(mean(samples$BUGSoutput$sims.list$nxbeta))
print(mean(samples$BUGSoutput$sims.list$nybeta))

# print(samples$BUGSoutput$sims.list$Sigma[1])
# print(dim(samples$BUGSoutput$sims.list$S))
# print(dim(samples$BUGSoutput$sims.list$CONST))
# humanbeta <- samples$BUGSoutput$sims.list$humanbeta
# nxbeta <- samples$BUGSoutput$sims.list$nxbeta
# nybeta <- samples$BUGSoutput$sims.list$nybeta
# save(llmbeta, humanbeta, nxbeta, nybeta, file = 'prompt_data_new.RData')