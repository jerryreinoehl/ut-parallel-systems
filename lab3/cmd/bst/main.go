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

type context struct {
	numHashWorkers uint
	numDataWorkers uint
	numCompWorkers uint
	trees          []*btree.BTree[int]
}

func main() {
	numHashWorkers := flag.Uint("hash-workers", 1, "Number of hash workers")
	numDataWorkers := flag.Uint("data-workers", 0, "Number of data workers")
	numCompWorkers := flag.Uint("comp-workers", 0, "Number of comparison workers")
	input := flag.String("input", "", "Input file path")
	flag.Parse()

	if *input == "" {
		log.Fatal("Must specify input")
	}

	trees := loadTrees(*input)

	ctx := context{*numHashWorkers, *numDataWorkers, *numCompWorkers, trees}

	if uint(*numDataWorkers) == 0 {
		hashTreesOnly(&ctx)
	} else {
		hashTreesMapped(&ctx)
	}
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

// Hash each bst and report hash time. Do not add hashes to map.
func hashTreesOnly(ctx *context) {
	ids := make(chan int, len(ctx.trees))

	var hashWg sync.WaitGroup
	hashStart := time.Now()
	var hashStop time.Time

	hashWg.Add(int(ctx.numHashWorkers))

	for i := uint(0); i < ctx.numHashWorkers; i++ {
		go func() {
			defer hashWg.Done()

			for id := range ids {
				_ = hash(ctx.trees[id])
			}
		}()
	}

	for i := range ctx.trees {
		ids <- i
	}
	close(ids)

	hashWg.Wait()
	hashStop = time.Now()

	fmt.Printf("hashTime: %f\n", hashStop.Sub(hashStart).Seconds())
}

func hashTreesMapped(ctx *context) {

}

func hash(bt *btree.BTree[int]) int {
	hash := 1
	bt.InOrderFunc(func(value int) {
		newValue := value + 2
		hash = (hash * newValue + newValue) % 1000
	})
	return hash
}
