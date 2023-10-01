library(ggplot2)

df <- read.csv("time.csv")

df_1e5 = df[df$threshold == 1e-5, ]
df_1e10 = df[df$threshold == 1e-10, ]

ggplot(df_thres_1e5, aes(factor(size), time, fill=impl)) + geom_bar(stat="identity", position="dodge")

gg_runtime_1e5 <- ggplot(
  df_1e5,
  aes(factor(size),
  time,
  fill=factor(impl, levels=c("seq", "cuda", "shmem", "thrust", "ideal")))
) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_manual(
    name="Implementation",
    label=c("seq", "cuda", "shmem", "thrust", "ideal"),
    values=c("maroon", "steelblue", "darkgreen", "orange", "#4d4dff")
  ) +
  labs(
    title="Total Runtime vs. Number of Points for each Implementation (Threshold 1e-5)",
    x="Number of Points",
    y="Runtime (ms)"
  ) +
  scale_y_continuous(breaks=seq(0, 1200, 100))

gg_runtime_1e10 <- ggplot(
  df_1e10,
  aes(factor(size),
  time,
  fill=factor(impl, levels=c("seq", "cuda", "shmem", "thrust", "ideal")))
) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_manual(
    name="Implementation",
    label=c("seq", "cuda", "shmem", "thrust", "ideal"),
    values=c("maroon", "steelblue", "darkgreen", "orange", "#4d4dff")
  ) +
  labs(
    title="Total Runime vs. Number of Points for each Implementation (Threshold 1e-10)",
    x="Number of Points",
    y="Time (ms)"
  ) +
  scale_y_continuous(breaks=seq(0, 4000, 200))

gg_itertime_1e10 <- ggplot(
  df_1e10[df_1e10$impl != "ideal", ],
  aes(factor(size),
  iter_time,
  fill=factor(impl, levels=c("seq", "cuda", "shmem", "thrust")))
) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_manual(
    name="Implementation",
    label=c("seq", "cuda", "shmem", "thrust"),
    values=c("maroon", "steelblue", "darkgreen", "orange")
  ) +
  labs(
    title="Time Per Iteration vs. Number of Points for each Implementation",
    x="Number of Points",
    y="Time Per Iteration (ms)"
  ) +
  scale_y_continuous(breaks=seq(0, 30, 2))

svg("time_1e-5.svg")
plot(gg_runtime_1e5)

svg("time_1e-10.svg")
plot(gg_runtime_1e10)

svg("iter_time_1e-10.svg")
plot(gg_itertime_1e10)

df <- read.csv("transfer.csv")

gg_transfer_time <- ggplot(
  df,
  aes(factor(impl),
  dt,
  fill=factor(impl, levels=c("cuda", "shmem", "thrust")))
) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_manual(
    name="Implementation",
    label=c("cuda", "shmem", "thrust"),
    values=c("steelblue", "darkgreen", "orange")
  ) +
  labs(
    title="Percent of Total Runtime Spend in Data Transfer",
    x="Implementation",
    y="Percent of Runtime"
  ) +
  scale_y_continuous(breaks=seq(0, 100, 10))

svg("transfer_time.svg")
plot(gg_transfer_time)
