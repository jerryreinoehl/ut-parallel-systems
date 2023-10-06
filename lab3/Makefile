GO := go
TARGET := bst
SOURCES = $(shell find . -iname '*.go')

$(TARGET): $(SOURCES)
	$(GO) build -o $@ ./cmd/bst

.PHONY: test
test:
	@./scripts/test

.PHONY: run
run:
	go run ./cmd/bst

.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: clean
clean:
	rm -f $(TARGET)
