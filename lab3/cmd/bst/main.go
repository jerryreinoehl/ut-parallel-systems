package main

import (
	"bst/pkg/btree"
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	numHashWorkers := flag.Uint("hash-workers", 1, "Number of hash workers")
	numDataWorkers := flag.Uint("data-workers", 0, "Number of data workers")
	numCompWorkers := flag.Uint("comp-workers", 0, "Number of comparison workers")
	input := flag.String("input", "", "Input file path")
	flag.Parse()

	// Silence unused variable errors for now.
	_ = numDataWorkers
	_ = numCompWorkers

	if *input == "" {
		log.Fatal("Must specify input")
	}

	trees := loadTrees(*input)

	type hashId struct { hash, id int }

	ids := make(chan int, len(trees))
	hashes := make(chan hashId, len(trees))
	var hashWg sync.WaitGroup

	hashWg.Add(int(*numHashWorkers))

	hashStart := time.Now()
	var hashStop time.Time
	for i := uint(0); i < *numHashWorkers; i++ {
		go func() {
			defer hashWg.Done()

			for id := range ids {
				hash := hash(trees[id])
				hashes <- hashId{hash, id}
			}
		}()
	}

	go func() {
		//for hashId := range hashes {
		for range hashes {
		}
	}()

	for i := range trees {
		ids <- i
	}
	close(ids)

	hashWg.Wait()
	hashStop = time.Now()
	close(hashes)

	fmt.Printf("hashGroupTime: %v\n", hashStop.Sub(hashStart))
}

func loadTrees(file string) []*btree.BTree[int] {
	trees := make([]*btree.BTree[int], 0)

	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}

	defer f.Close()

	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		tree := btree.NewBTree[int]()
		trees = append(trees, &tree)

		line := scanner.Text()

		for _, item := range strings.Split(line, " ") {
			parsed, err := strconv.Atoi(item)
			if err != nil {
				log.Fatal(err)
			}
			tree.Insert(parsed)
		}
	}

	return trees
}

func hash(bt *btree.BTree[int]) int {
	hash := 1
	bt.InOrderFunc(func(value int) {
		newValue := value + 2
		hash = (hash * newValue + newValue) % 1000
	})
	return hash
}
