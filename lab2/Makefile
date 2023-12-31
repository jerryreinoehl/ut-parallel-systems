CC := nvcc
CFLAGS := -arch sm_60 -O3 -Xcompiler -Wall,-Werror

OBJS = main.o args.o kmeans.o kmeans_cuda.o kmeans_shmem.o kmeans_thrust.o
SRCS = main.cpp args.cpp kmeans.cpp kmeans_cuda.cu kmeans_shmem.cu kmeans_thrust.cu
HEADERS = src/cudaptr.h src/cudaarena.h src/cudavect.cuh
TARGET := kmeans

SUBMISSION := gerald_reinoehl_ger846_lab2.tar.gz

.PHONY: all
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: src/%.cpp src/%.h $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: src/%.cu src/%.h $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: src/%.cu src/%.cuh $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: debug
debug: CFLAGS = -arch sm_60 -g -O0 -Xcompiler -Wall,-Werror,-Wextra
debug: $(TARGET)

.PHONY: test
test: CFLAGS = -arch sm_60 -g -O0 -Xcompiler -Wall
test: $(TARGET)

.PHONY: prof
prof: CFLAGS += -Xcompiler -pg
prof: $(TARGET)

.PHONY: submission
submission: clean
	tar -czvf $(SUBMISSION) \
		--exclude='test/*' --exclude=$(SUBMISSION) --exclude-vcs \
		--transform 's|^|gerald_reinoehl_ger846_lab2/|' \
		*

.PHONY: clean
clean:
	rm -f *.o $(TARGET)
