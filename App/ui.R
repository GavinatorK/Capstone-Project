# ui.R

shinyUI(fluidPage(
 
   headerPanel(""),
  
      titlePanel("Special Healthcare Needs App"),

  sidebarLayout(
    sidebarPanel(
      helpText("Create demographic maps with 
               information from NIS CSHCN Surverys."),
      conditionalPanel(
        "$('li.active a').first().html()==='Map Charts'",
       
      selectInput("var", 
                      label = "Choose a Condition to display",
                  choices = c("ASTHMA",	"ATTENTION.DEFICIT.DISORDER",	"AUTISM",	
                              "DOWN.SYNDROME","MENTAL.RETARDATION",	"EMOTIONAL.PROBLEMS","DIABETES",
                              "CHILD.USES.INSULIN",	"HEART.PROBLEM",	"BLOOD.PROBLEMS",	"CYSTIC.FIBROSIS",
                              "CEREBRAL.PALSY",	"MUSCULAR.DYSTROPHY","SEIZURE.DISORDER",
                              "MIGRAINE.OR.FREQUENT.HEADACHES",	"JOINT.PROBLEMS",	"ALLERGIES",	"FOOD.ALLERGIES"),
                  selected = "ASTHMA")),
      conditionalPanel(
        "$('li.active a').first().html()==='National'",
        
        selectInput("varn", 
                    label = "Choose a Variable to display",
                    choices = c("Comorbidity", "Dissatisfaction",
                                "Dissat by loc", "Dissat by Condition"),
                    selected = "Dissatisfaction"),
        conditionalPanel(
          condition = "input.varn == 'Comorbidity'",
        sliderInput("comor", 
                    label = "Comorbid Conditions 1 - 5 or more",
                    min = 1, max = 5, value = 1, dragRange=FALSE, step=1, sep="")
        )
      ),
      conditionalPanel(
        "$('li.active a').first().html()==='State'",
        
        selectInput("vars", 
                    label = "Choose a state to display",
                    choices = c("Alaska",         "Alabama",        "Arkansas",       "Arizona",        "California",    
                                "Colorado",       "Connecticut",    "Washington DC",  "Delaware",       "Florida",       
                                "Georgia",        "Hawaii",         "Iowa",           "Idaho",          "Illinois",      
                                "Indiana",        "Kansas",         "Kentucky",       "Louisiana",      "Massachusetts", 
                                "Maryland",       "Maine",          "Michigan",       "Minnesota",      "Missouri",      
                                "Mississippi",    "Montana",        "North Carolina", "North Dakota",   "Nebraska",      
                                "New Hampshire",  "New Jersey",     "New Mexico",     "Nevada",         "New York ",     
                                "Ohio",           "Oklahoma",       "Oregon",         "Pennsylvania",   "Rhode Island",  
                                "South Carolina", "South Dakota",   "Tennessee",      "Texas",          "Utah",          
                                "Virginia",       "Vermont",        "Washington",     "Wisconsin",      "West Virginia", 
                                "Wyoming"),
                    selected = "Alabama")),
      sliderInput("S_Year", 
                  label = "Year of Interest:",
                  min = 2001, max = 2010, value = 2006, dragRange=FALSE, step=5, sep="")
      ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Map Charts", plotOutput("map")), 
        tabPanel("National", plotOutput("national")), 
        tabPanel("State", plotOutput("state"))
      )
    )
  )
))