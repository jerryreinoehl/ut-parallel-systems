CC := g++
CFLAGS := -std=c++17 -O2 -Wall -Werror -Wextra
SRC = ./src/*.cpp
OBJ = main.o args.o particle.o spatialpartitiontree.o vector2d.o vector3d.o mpi.o
INC = -I/src/ -I/opt/mpich/include
LIB = -L/opt/mpich/lib/ -lmpi

TARGET := nbody

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LIB) $(INC) -o $@ $^

%.o: src/%.cpp
	$(CC) $(CFLAGS) $(INC) -c -o $@ $^

%.o: src/%.cpp src/%.h
	$(CC) $(CFLAGS) $(INC) -c -o $@ $^

.PHONY: debug
debug: CFLAGS = -g -O0 -fsanitize=address
debug: $(TARGET)

.PHONY: visual
visual: CFLAGS += -DVISUAL
visual: OBJ += visual.o
visual: LIB += -lGL -lGLEW -lglfw -lglut
visual: $(OBJ) visual.o
	$(CC) $(CFLAGS) $(LIB) $(INC) -o $(TARGET) $^

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)
