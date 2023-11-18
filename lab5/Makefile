CC := g++
CFLAGS := -std=c++17 -O2 -Wall -Werror -Wextra
SRC = ./src/*.cpp
OBJ = main.o args.o particle.o
INC = ./src/

TARGET := nbody

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -I$(INC) -o $@ $^

%.o: src/%.cpp
	$(CC) $(CFLAGS) -I$(INC) -c -o $@ $^

%.o: src/%.cpp src/%.h
	$(CC) $(CFLAGS) -I$(INC) -c -o $@ $^

.PHONY: debug
debug: CFLAGS = -g -O0
debug: $(TARGET)

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)
