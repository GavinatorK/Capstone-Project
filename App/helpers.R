# Note: percent map is designed to work with the counties data set
# It may not work correctly with other data sets if their row order does 
# not exactly match the order in which the maps package plots counties
percent_map <- function(var, color, legend.title, min, max) {

  # generate vector of fill colors for map
  shades <- colorRampPalette(c("white", color))(10)
  
  # constrain gradient to percents that occur between min and max
  var <- pmax(var, min)
  var <- pmin(var, max)
  percents <- as.integer(cut(var, 10, 
    include.lowest = TRUE, ordered = TRUE))
  fills <- shades[percents]
  
  
  # overlay state borders
  map("state", fill = TRUE, col = fills, 
      resolution = 0, lty = 0, projection = "polyconic", 
      myborder = 0, mar = c(0,0,0,0))
  
  # add a legend
  if (legend.title!="Disatisfaction proprtion"){
  inc <- as.integer((max - min) / 4)
  legend.text <- c(paste0(min, " cases  "),
    paste0(min + 2 * inc, " cases  "),
    paste0(min + 3 * inc, " cases  "),
    paste0(min + 4 * inc, " cases  "),
    paste0(max, " cases")) 
  }
  else
  {
    inc <- as.numeric((max - min) / 5)
    legend.text <- c(paste0(round(min, digits=2), " percent"),
                     paste0(round(min + 2 * inc, digits=2), " percent"),
                     paste0(round(min + 3 * inc, digits=2), " percent"),
                     paste0(round(min + 4 * inc, digits=2), " percent"),
                     paste0(round(max, digits=2), " percent")) 
    
  }
  
  legend("bottomleft", 
    legend = legend.text,
    inset=0.0,
    fill = shades[c(1,3,5,7,10)], 
    title = legend.title,lty=0
    ,cex=0.75)
    
}