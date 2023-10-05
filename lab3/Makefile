GO := go
TARGET := bst
SOURCES = $(shell find . -iname '*.go')

$(TARGET): $(SOURCES)
	$(GO) build -o $@ ./cmd/bst

.PHONY: clean
clean:
	rm -f $(TARGET)
