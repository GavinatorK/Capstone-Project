# server.R

library(maps)
library(mapproj)
library(ggplot2)
require(rNVD3)
dat <- read.csv("data/conds.csv")
disdat<-read.csv("data/2006_dissat.csv")
dissatloc<-read.csv("data/2006_disbyloc.csv")
condst<-read.csv("data/2006_dissatisbycond.csv")
discomor<-read.csv("data/2006_discomorb.csv")
source("helpers.R")
shinyServer(
  function(input, output) {
    output$map <- renderPlot({
      mydat=dat[dat$Year==input$S_Year,]
      args <- switch(input$var,
                     "ASTHMA" = list(mydat$ASTHMA, "brown", "ASTHMA CASES", min(dat$ASTHMA), max(dat$ASTHMA)),
                     "ATTENTION.DEFICIT.DISORDER" = list(mydat$ATTENTION.DEFICIT.DISORDER, "brown", "ATTENTION.DEFICIT.DISORDER", min(dat$ATTENTION.DEFICIT.DISORDER), max(dat$ATTENTION.DEFICIT.DISORDER)),
                     "AUTISM" = list(mydat$AUTISM, "brown", "AUTISM", min(dat$AUTISM), max(dat$AUTISM)),
                     "DOWN.SYNDROME" = list(mydat$DOWN.SYNDROME, "brown", "DOWN.SYNDROME", min(dat$DOWN.SYNDROME),max(dat$DOWN.SYNDROME)),
                     "MENTAL.RETARDATION"=list(mydat$MENTAL.RETARDATION, "brown", "MENTAL.RETARDATION", min(dat$MENTAL.RETARDATION),max(dat$MENTAL.RETARDATION)),
                     "EMOTIONAL.PROBLEMS"=list(mydat$EMOTIONAL.PROBLEMS, "brown", "EMOTIONAL.PROBLEMS", min(dat$EMOTIONAL.PROBLEMS),max(dat$EMOTIONAL.PROBLEMS)),
                     "DIABETES"=list(mydat$DIABETES, "brown", "DIABETES", min(dat$DIABETES),max(dat$DIABETES)),
                     "CHILD.USES.INSULIN"=list(mydat$CHILD.USES.INSULIN, "brown", "CHILD.USES.INSULIN", min(dat$CHILD.USES.INSULIN),max(dat$CHILD.USES.INSULIN)),
                     "HEART.PROBLEM"=list(mydat$HEART.PROBLEM, "brown", "HEART.PROBLEM", min(dat$HEART.PROBLEM),max(dat$HEART.PROBLEM)),
                     "BLOOD.PROBLEMS"=list(mydat$BLOOD.PROBLEMS, "brown", "BLOOD.PROBLEMS", min(dat$BLOOD.PROBLEMS),max(dat$BLOOD.PROBLEMS)),
                     "CYSTIC.FIBROSIS"=list(mydat$CYSTIC.FIBROSIS, "brown", "CYSTIC.FIBROSIS", min(dat$CYSTIC.FIBROSIS),max(dat$CYSTIC.FIBROSIS)),
                     "CEREBRAL.PALSY"=list(mydat$CEREBRAL.PALSY, "brown", "CEREBRAL.PALSY", min(dat$CEREBRAL.PALSY),max(dat$CEREBRAL.PALSY)),
                     "MUSCULAR.DYSTROPHY"=list(mydat$MUSCULAR.DYSTROPHY, "brown", "MUSCULAR.DYSTROPHY", min(dat$MUSCULAR.DYSTROPHY),max(dat$MUSCULAR.DYSTROPHY)),
                     "SEIZURE.DISORDER"=list(mydat$SEIZURE.DISORDER, "brown", "SEIZURE.DISORDER", min(dat$SEIZURE.DISORDER),max(dat$SEIZURE.DISORDER)),
                     "MIGRAINE.OR.FREQUENT.HEADACHES"=list(mydat$MIGRAINE.OR.FREQUENT.HEADACHES, "brown", "MIGRAINE.OR.FREQUENT.HEADACHES", min(dat$MIGRAINE.OR.FREQUENT.HEADACHES),max(dat$MIGRAINE.OR.FREQUENT.HEADACHES)),
                     "JOINT.PROBLEMS"=list(mydat$JOINT.PROBLEMS, "brown", "JOINT.PROBLEMS", min(dat$JOINT.PROBLEMS),max(dat$JOINT.PROBLEMS)),	
                     "ALLERGIES"=list(mydat$ALLERGIES, "brown", "ALLERGIES", min(dat$ALLERGIES),max(dat$ALLERGIES)),
                     "FOOD.ALLERGIES"=list(mydat$FOOD.ALLERGIES, "brown", "FOOD.ALLERGIES", min(dat$FOOD.ALLERGIES),max(dat$FOOD.ALLERGIES))
      )
      
      args$min <- input$range[1]
      args$max <- input$range[2]
      
      do.call(percent_map, args)
    },height = 600, width =700)
# Functions for tabs in National
    datasetInput <- reactive({
      switch(input$varn,
             "Dissatisfaction" = disdat,
             "Dissat by loc" = dissatloc,
             "Comorbidity" = discomor,
             "Others" =OTHERS)
    })
    output$national <- renderPlot({
      d1=datasetInput()
      dy1=d1[d1$Year==input$S_Year,]
      switch(input$varn,
      
     "Dissatisfaction"= ggplot(data=dy1, aes(x=NAME, y=Percent, fill="red")) +
        geom_bar(stat="identity", fill="#56B4E9")+theme(text=element_text(size=15),legend.position="none",axis.text.x = element_text(angle = 90, hjust = 1))+
      xlab("STATE") + ylab("Percent") +
        ggtitle("Dissatisfaction Percent with Healthcare Services"),
      
      
      
       "Dissat by loc"= ggplot(data=dy1, aes(x=NAME, y=Percent, fill=MSASTATR)) +
          geom_bar(stat="identity")+theme(text=element_text(size=15),legend.position="left",axis.text.x = element_text(angle = 90, hjust = 1))+
          xlab("STATE") + ylab("Proportion") +scale_fill_manual(values=c("maroon2", "lightslateblue","olivedrab1"),guide_legend(title = "Location"))+
          ggtitle("Dissatisfaction Percent with Healthcare Services"),
      
      "Comorbidity"=do.call(percent_map, list(dy1[dy1$comorb==input$comor,]$Proportion,"Orange", "Disatisfaction proprtion", min(dy1[dy1$comorb==input$comor,]$Proportion), max(dy1[dy1$comorb==input$comor,]$Proportion)))

   ) },height = 600, width =700)
  


#Dissatisfaction percents by condition for each state
# Functions for tabs in National

    output$state <- renderPlot({
      cs1=condst[condst$Year==input$S_Year,]
      cs2=cs1[cs1$NAME==input$vars,]
             
              ggplot(data=cs2, aes(x=variable, y=value, fill="red")) +
               geom_bar(stat="identity", fill="darkgoldenrod3")+theme(text=element_text(size=15),legend.position="none",axis.text.x = element_text(angle = 90, hjust = 1))+
               xlab("Condition") + ylab("Percent") +
               ggtitle("Dissatisfaction Percent with Healthcare Services by Condition")
    
            
     },height = 600, width =700)
    
  }
)
