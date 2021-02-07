library(class)
library(gmodels)
library(caret)
library(tidyverse)
library(tokenizers)
library(tidytext)
library(wordcloud)
library(tm)
library(dplyr)
library(igraph)
library(ggraph)
library(stringr)
library(caret)
library(naivebayes)
dataset=read.csv(file = "/Users/nselvarajan/Desktop/test/archive/cleaned.csv")
head(dataset)

dataset$positive = as.factor(dataset$stars > 3)

myCorpus <- Corpus(VectorSource(dataset$text))
corpus <- tm_map(myCorpus,removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, stemDocument, language = 'english')
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, stripWhitespace)


rating5 <- subset(dataset, stars == "5") 
myCorpusRating5 <- Corpus(VectorSource(rating5$text))
myCorpusRating5 <- tm_map(myCorpusRating5,removeNumbers)
myCorpusRating5 <- tm_map(myCorpusRating5, removePunctuation)
myCorpusRating5 <- tm_map(myCorpusRating5, tolower)
myCorpusRating5 <- tm_map(myCorpusRating5, stemDocument, language = 'english')
myCorpusRating5 <- tm_map(myCorpusRating5, removeWords, stopwords('english'))
myCorpusRating5 <- tm_map(myCorpusRating5, stripWhitespace)
bag_of_words_rating_5 <- DocumentTermMatrix(myCorpusRating5)    
##creating DTM to get frequencies
inspect(bag_of_words_rating_5)
dataframeRating5<-data.frame(text=unlist(sapply(myCorpusRating5, `[`)), stringsAsFactors=F) ##creating data fram from matrix 
y<-head(dataframeRating5,100)
# word cloud visualization
library(wordcloud)
wordcloud(y$text)




bag_of_words <- DocumentTermMatrix(corpus)  ##creating DTM to get frequencies
inspect(bag_of_words)

dataframe<-data.frame(text=unlist(sapply(corpus, `[`)), stringsAsFactors=F) ##creating data fram from matrix 
dataset$text <- dataframe$text
y<-head(dataset,100)

# word cloud visualization
library(wordcloud)
wordcloud(y$text)

summary(dataset)
nrow(dataset)
head(dataset)

dataset_train <- dataset[1:24000,]      ###dividing data into training and test set
dataset_test <- dataset[24000:331400,]
myCorpus_model_train <- Corpus(VectorSource(dataset_train$text)) ##creating corpus for training

dtm_train <- DocumentTermMatrix(myCorpus_model_train) ##since this data was already cleaned before, we can straigtaway move to DTM
dtm_train <- removeSparseTerms(dtm_train,0.95)

myCorpus_model_test <- Corpus(VectorSource(dataset_test$text)) ##creating corpus for test

dtm_test <- DocumentTermMatrix(myCorpus_model_test) ##since this data was already cleaned before, we can straigtaway move to DTM
dtm_test <- removeSparseTerms(dtm_test,0.95)


model <- naive_bayes(as.data.frame(as.matrix(dtm_train)), dataset_train$positive, laplace = 1)
model

model_predict <- predict(model, as.data.frame(as.matrix(dtm_test)))
confusionMatrix(model_predict, dataset_test$positive)


library(gmodels)
CrossTable(model_predict, dataset_test$positive,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

library(Rgraphviz)
findFreqTerms(dtm_train, lowfreq = 25)
