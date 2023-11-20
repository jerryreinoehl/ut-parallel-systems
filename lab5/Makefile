CC := g++
CFLAGS := -std=c++17 -O2 -Wall -Werror -Wextra
SRC = ./src/*.cpp
OBJ = main.o args.o particle.o spatialpartitiontree.o vector2d.o vector3d.o
INC = ./src/
LIB = -lGL -lGLEW -lglfw -lglut

TARGET := nbody

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LIB) -I$(INC) -o $@ $^

%.o: src/%.cpp
	$(CC) $(CFLAGS) -I$(INC) -c -o $@ $^

%.o: src/%.cpp src/%.h
	$(CC) $(CFLAGS) -I$(INC) -c -o $@ $^

.PHONY: debug
debug: CFLAGS = -g -O0 -fsanitize=address
debug: $(TARGET)

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)
