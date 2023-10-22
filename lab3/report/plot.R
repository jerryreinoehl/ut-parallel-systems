library(ggplot2)

options(scipen=999)

hashtimecoarse <- read.csv("hashtimecoarse.csv")
hashtimefine <- read.csv("hashtimefine.csv")

gghashtimecoarse <- ggplot(hashtimecoarse, aes(x=threads, y=time)) +
  geom_point(size=3, color="steelblue") +
  geom_line(linewidth=1, color="steelblue") +
  scale_x_continuous(breaks=seq(0, 100, 4), limits=c(0, 100)) +
  scale_y_continuous(breaks=seq(0, 35, 1), limits=c(0, NA)) +
  labs(
    title="Hash Time vs Number of Goroutines (coarse.txt)",
    x="Number of Goroutines",
    y="Hash Time (ms)"
  )

svg("hashtimecoarse.svg")
plot(gghashtimecoarse)

hashtimefine <- hashtimefine[hashtimefine$threads < 100,]

gghashtimefine <- ggplot(hashtimefine, aes(x=threads, y=time)) +
  geom_point(size=3, color="maroon") +
  geom_line(linewidth=1, color="maroon") +
  #geom_hline(yintercept=hashtimefinehline, linetype="dashed", color="maroon") +
  scale_x_continuous(breaks=seq(0, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 50, 5), limits=c(0, NA)) +
  labs(
    title="Hash Time vs Number of Goroutines (fine.txt)",
    x="Number of Goroutines",
    y="Hash Time (ms)"
  )

svg("hashtimefine.svg")
plot(gghashtimefine)

hashtimefinelarge <- read.csv("hashtimefinelarge.csv")
hashtimefineseqtime = hashtimefine[hashtimefine$threads == 1,]$time

gghashtimefinelarge <- ggplot(hashtimefinelarge, aes(x=threads, y=time)) +
  geom_point(size=3, color="maroon") +
  geom_line(linewidth=1, color="maroon") +
  scale_x_continuous(breaks=seq(0, 100000, 10000)) +
  scale_y_continuous(breaks=seq(0, 270, 10), limits=c(0, NA)) +
  labs(
    title="Hash Time vs Number of Goroutines (fine.txt)",
    x="Number of Goroutines",
    y="Hash Time (ms)"
  )

svg("hashtimefinelarge.svg")
plot(gghashtimefinelarge)

hashgrouptimecoarse <- read.csv("hashgrouptimecoarse.csv")

gghashgrouptimecoarse <- ggplot(hashgrouptimecoarse, aes(x=threads)) +
  geom_point(aes(y=channel_time, color="channels"), size=3) +
  geom_line(aes(y=channel_time, color="channels"), linewidth=1) +

  geom_point(aes(y=mutex_time, color="mutex/semaphore"), size=3) +
  geom_line(aes(y=mutex_time, color="mutex/semaphore"), linewidth=1) +

  scale_x_continuous(breaks=seq(0, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 15, 1), limits=c(0, NA)) +
  labs(
    title="Hash Group Time vs Number of Data Workers (coarse.txt)",
    x="Number of Data Workers",
    y="Hash Group Time (ms)"
  ) +
  scale_color_manual(
    breaks=c("channels", "mutex/semaphore"),
    values=c("channels"="steelblue", "mutex/semaphore"="maroon")
  ) +
  theme(
    legend.title=element_blank(),
    legend.position=c(0.5, 0.5)
  )

svg("hashgrouptimecoarse.svg")
plot(gghashgrouptimecoarse)
