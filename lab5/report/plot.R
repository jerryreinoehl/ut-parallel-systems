library(ggplot2)

comptime <- data.frame("Category" = c("Tree Construction", "Center of Mass Computation", "Force Calculation"),
                       "Time" = c(50.005494, 20.460199, 1011.8566))

ggcomptime <- ggplot(comptime, aes(x="", y=Time, fill=factor(
  Category, levels=c("Force Calculation", "Center of Mass Computation", "Tree Construction")))) +
  geom_bar(stat="identity", width=0.5) +
  coord_polar("y", start=0) +
  labs(
    title="Loop Execution Time Breakdown (100,000 Particles)",
    x="",
    y="Runtime (ms)"
  ) + 
  theme(legend.title=element_blank()) +
  scale_y_continuous(breaks=seq(0, 1200, 50)) +
  #geom_text(aes(label = paste0(Time, "ms")), position = position_stack(vjust=0.5), angle = c(10, 10, 10), size = 3) +
  scale_fill_manual(values=c("maroon", "steelblue", "darkgreen"))

svg("comptime.svg")
plot(ggcomptime)
