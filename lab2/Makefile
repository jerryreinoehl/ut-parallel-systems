CC := nvcc
CFLAGS := -O3 -Xcompiler -Wall,-Werror

OBJS = main.o args.o kmeans.o
SRCS = main.cpp args.cpp kmeans.cpp
TARGET := kmeans

.PHONY: all
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: src/%.cpp
	$(CC) $(CFLAGS) -c -o $@ $^

.PHONY: debug
debug: CFLAGS = -g -O0 -Xcompiler -Wall,-Werror,-fsanitize=address
debug: $(TARGET)

.PHONY: prof
prof: CFLAGS += -Xcompiler -pg
prof: $(TARGET)

.PHONY: clean
clean:
	rm *.o $(TARGET)
