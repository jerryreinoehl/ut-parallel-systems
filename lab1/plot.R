library(ggplot2)

pthread <- read.csv("pthread-2.csv")
pthread[2:5] <- pthread[2:5] / 1000000
pthread_baseline <- pthread[1, 2:ncol(pthread)]  # Sequential times
pthread <- pthread[-1:-2,]                       # Remove first two rows
pthread$speedup1k <- pthread_baseline$X1k / pthread$X1k
pthread$speedup8k <- pthread_baseline$X8k / pthread$X8k
pthread$speedup16k <- pthread_baseline$X16k / pthread$X16k

pthread_l10 <- read.csv("pthread-l10.csv")
pthread_seq_time_16k_l10 <- pthread_l10$X16k[1] / 1000000
pthread$X16k_l10 <- pthread_l10$X16k[3:nrow(pthread_l10)] / 1000000

pthread_l65 <- read.csv("pthread-l65.csv")
pthread_seq_time_16k_l65 <- pthread_l65$X16k[1] / 1000000
pthread$X16k_l65 <- pthread_l65$X16k[3:nrow(pthread_l65)] / 1000000

pthread_l100 <- read.csv("pthread-l100.csv")
pthread_seq_time_16k_l100 <- pthread_l100$X16k[1] / 1000000
pthread$X16k_l100 <- pthread_l100$X16k[3:nrow(pthread_l100)] / 1000000

gg1k <- ggplot(pthread, aes(x=threads)) +
  geom_point(aes(y=X1k), size=3, color="steelblue") +
  geom_line(aes(y=X1k), linewidth=1, color="steelblue") +
  geom_hline(yintercept=pthread_baseline[1, 1], linetype="dashed") +
  labs(
    title="Execution Time for 1K Elements (pthread_barrier_t)",
    x="Number of Threads",
    y="Execution Time (s)"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 0.24, 0.02)) +
  theme(text=element_text(size=18))

gg8k <- ggplot(pthread, aes(x=threads)) +
  geom_point(aes(y=X8k), size=3, color="maroon") +
  geom_line(aes(y=X8k), linewidth=1, color="maroon") +
  geom_hline(yintercept=pthread_baseline[1, 2], linetype="dashed") +
  labs(
    title="Execution Time for 8K Elements (pthread_barrier_t)",
    x="Number of Threads",
    y="Execution Time (s)"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 1.8, 0.1)) +
  theme(text=element_text(size=18))

gg16k <- ggplot(pthread, aes(x=threads)) +
  geom_point(aes(y=X16k), size=3, color="darkgreen") +
  geom_line(aes(y=X16k), linewidth=1, color="darkgreen") +
  geom_hline(yintercept=pthread_baseline[1, 3], linetype="dashed") +
  labs(
    title="Execution Time for 16K Elements (pthread_barrier_t)",
    x="Number of Threads",
    y="Execution Time (s)"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 3.2, 0.2)) +
  theme(text=element_text(size=18))

ggspeedup <- ggplot(pthread, aes(x=threads), height=200, width=100) +
  geom_line(aes(x=seq(2, 32, 2), y=seq(2, 32, 2)), linewidth=0.5, linetype="dashed") +

  geom_point(aes(y=speedup1k, color="1k"), size=2) +
  geom_line(aes(y=speedup1k, color="1k"), linewidth=0.5) +

  geom_point(aes(y=speedup8k, color="8k"), size=2) +
  geom_line(aes(y=speedup8k, color="8k"), linewidth=0.5) +

  geom_point(aes(y=speedup16k, color="16k"), size=2) +
  geom_line(aes(y=speedup16k, color="16k"), linewidth=0.5) +

  scale_color_manual(
    breaks=c("1k", "8k", "16k"),
    values=c("1k"="steelblue", "8k"="maroon", "16k"="darkgreen")
  ) +
  labs(
    title="Speedup 100,000 Loops (pthread_barrier_t)",
    x="Number of Threads",
    y="Speedup"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 32, 2), limits=c(0, 10)) +
  theme(text=element_text(size=10)) +
  theme(legend.title=element_blank()) +
  coord_fixed()

gg16k_l10 <- ggplot(pthread, aes(x=threads)) +
  geom_point(aes(y=X16k_l10, color="10"), size=3) +
  geom_line(aes(y=X16k_l10, color="10"), linewidth=1) +
  geom_hline(yintercept=pthread_seq_time_16k_l10, linetype="dashed", color="steelblue") +

  geom_point(aes(y=X16k_l65, color="65"), size=3) +
  geom_line(aes(y=X16k_l65, color="65"), linewidth=1) +
  geom_hline(yintercept=pthread_seq_time_16k_l65, linetype="dashed", color="maroon") +

  geom_point(aes(y=X16k_l100, color="100"), size=3) +
  geom_line(aes(y=X16k_l100, color="100"), linewidth=1) +
  geom_hline(yintercept=pthread_seq_time_16k_l100, linetype="dashed", color="darkgreen") +

  scale_color_manual(
    name="loops",
    breaks=c("10", "65", "100"),
    values=c("10"="steelblue", "65"="maroon", "100"="darkgreen")
  ) +
  labs(
    title="Execution Time for 16K Elements (pthread_barrier_t)",
    x="Number of Threads",
    y="Execution Time (s)"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 0.01, 0.001)) +
  theme(text=element_text(size=12))



svg("report/pthread-time-1k.svg")
plot(gg1k)

svg("report/pthread-time-8k.svg")
plot(gg8k)

svg("report/pthread-time-16k.svg")
plot(gg16k)

svg("report/pthread-speedup.svg")
plot(ggspeedup)

svg("report/pthread-time-16k-l10.svg")
plot(gg16k_l10)

spin <- read.csv("spin-3.csv")
spin_sleep <- read.csv("spin-sleep.csv")
spin_both <- read.csv("spin-both.csv")
spin_pthread <- read.csv("pthread-2.csv")

spin_df <- spin[1]
spin_df$spin <- spin$X16k
spin_df$sleep <- spin_sleep$X16k
spin_df$both <- spin_both$X16k
spin_df$pthread <- spin_pthread$X16k

spin_seq_time <- spin_df$spin[1] / 1000000
spin_df <- spin_df[-1:-2,]
spin_df[2:ncol(spin_df)] <- spin_df[2:ncol(spin_df)] / 1000000
spin_df$both_speedup1k <- spin$X1k[1] / spin_both$X1k[3:18]
spin_df$both_speedup8k <- spin$X8k[1] / spin_both$X8k[3:18]
spin_df$both_speedup16k <- spin$X16k[1] / spin_both$X16k[3:18]

spin_l10 <- read.csv("spin-l10.csv")
spin_seq_time_16k_l10 <- spin_l10$X16k[1] / 1000000
spin_df$X16k_l10 <- spin_l10$X16k[3:nrow(spin_l10)] / 1000000

spin_l65 <- read.csv("spin-l65.csv")
spin_seq_time_16k_l65 <- spin_l65$X16k[1] / 1000000
spin_df$X16k_l65 <- spin_l65$X16k[3:nrow(spin_l65)] / 1000000

spin_l200 <- read.csv("spin-l200.csv")
spin_seq_time_16k_l200 <- spin_l200$X16k[1] / 1000000
spin_df$X16k_l200 <- spin_l200$X16k[3:nrow(spin_l200)] / 1000000


ggspin_time_impl <- ggplot(spin_df, aes(x=threads)) +
  geom_hline(yintercept=spin_seq_time, linetype="dashed") +

  geom_point(aes(y=pthread, color="pthread"), size=2) +
  geom_line(aes(y=pthread, color="pthread"), linewidth=0.5) +

  geom_point(aes(y=sleep, color="sleep"), size=2) +
  geom_line(aes(y=sleep, color="sleep"), linewidth=0.5) +

  geom_point(aes(y=both, color="both"), size=2) +
  geom_line(aes(y=both, color="both"), linewidth=0.5) +

  geom_point(aes(y=spin, color="spin"), size=2) +
  geom_line(aes(y=spin, color="spin"), linewidth=0.5) +

  scale_color_manual(
    name="impl",
    breaks=c("spin", "sleep", "both", "pthread"),
    values=c("spin"="steelblue", "sleep"="maroon", "both"="darkgreen", "pthread"="orange")
  ) +
  labs(
    title="SpinBarrier Spin vs. Sleep vs. Both (16K Elements)",
    x="Number of Threads",
    y="Execution Time (s)"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 3.5, 0.1)) +
  theme(text=element_text(size=12))

svg("report/spin-time-impl.svg")
plot(ggspin_time_impl)

ggspin_speedup <- ggplot(spin_df, aes(x=threads), height=200, width=100) +
  geom_line(aes(x=seq(2, 32, 2), y=seq(2, 32, 2)), linewidth=0.5, linetype="dashed") +

  geom_point(aes(y=both_speedup1k, color="1k"), size=2) +
  geom_line(aes(y=both_speedup1k, color="1k"), linewidth=0.5) +

  geom_point(aes(y=both_speedup8k, color="8k"), size=2) +
  geom_line(aes(y=both_speedup8k, color="8k"), linewidth=0.5) +

  geom_point(aes(y=both_speedup16k, color="16k"), size=2) +
  geom_line(aes(y=both_speedup16k, color="16k"), linewidth=0.5) +

  scale_color_manual(
    breaks=c("1k", "8k", "16k"),
    values=c("1k"="steelblue", "8k"="maroon", "16k"="darkgreen")
  ) +
  labs(
    title="Speedup 100,000 Loops (SpinBarrier)",
    x="Number of Threads",
    y="Speedup"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  scale_y_continuous(breaks=seq(0, 32, 2), limits=c(0, 10)) +
  theme(text=element_text(size=10)) +
  theme(legend.title=element_blank()) +
  coord_fixed()

svg("report/spin-speedup.svg")
plot(ggspin_speedup)

ggspin_16k_l10 <- ggplot(spin_df, aes(x=threads)) +
  geom_point(aes(y=X16k_l10, color="10"), size=3) +
  geom_line(aes(y=X16k_l10, color="10"), linewidth=1) +
  geom_hline(yintercept=spin_seq_time_16k_l10, linetype="dashed", color="steelblue") +

  geom_point(aes(y=X16k_l65, color="65"), size=3) +
  geom_line(aes(y=X16k_l65, color="65"), linewidth=1) +
  geom_hline(yintercept=spin_seq_time_16k_l65, linetype="dashed", color="maroon") +

  geom_point(aes(y=X16k_l200, color="200"), size=3) +
  geom_line(aes(y=X16k_l200, color="200"), linewidth=1) +
  geom_hline(yintercept=spin_seq_time_16k_l200, linetype="dashed", color="darkgreen") +

  scale_color_manual(
    name="loops",
    breaks=c("10", "65", "200"),
    values=c("10"="steelblue", "65"="maroon", "200"="darkgreen")
  ) +
  labs(
    title="Execution Time for 16K Elements (SpinBarrier)",
    x="Number of Threads",
    y="Execution Time (s)"
  ) +
  scale_x_continuous(breaks=seq(2, 32, 2)) +
  #scale_y_continuous(breaks=seq(0, 0.01, 0.001)) +
  theme(text=element_text(size=12))


svg("report/spin-time-16k-l10.svg")
plot(ggspin_16k_l10)
