package main

import (
	"bst/pkg/btree"
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
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
	addWithMutex   bool
	trees          []*btree.BTree[int]
	hashGroups     map[int][]int
}

func main() {
	numHashWorkers := flag.Uint("hash-workers", 1, "Number of hash workers")
	numDataWorkers := flag.Uint("data-workers", 0, "Number of data workers")
	numCompWorkers := flag.Uint("comp-workers", 0, "Number of comparison workers")
	addWithMutex := flag.Bool(
		"add-with-mutex",
		false,
		"When specified use a mutex when adding to hash group map, otherwise use channel",
	)
	input := flag.String("input", "", "Input file path")
	flag.Parse()

	if *input == "" {
		log.Fatal("Must specify input")
	}

	trees := loadTrees(*input)

	ctx := context{
		*numHashWorkers,
		*numDataWorkers,
		*numCompWorkers,
		*addWithMutex,
		trees,
		make(map[int][]int),
	}

	if uint(*numDataWorkers) == 0 {
		hashTreesOnly(&ctx)
	} else {
		if *addWithMutex {
			hashTreesMappedMutex(&ctx)
		} else {
			hashTreesMappedChannel(&ctx)
		}
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

func hashTreesMappedChannel(ctx *context) {
	type hashId struct {hash, id int}

	ids := make(chan int, len(ctx.trees))
	hashes := make(chan hashId, len(ctx.trees))
	done := make(chan struct{})

	var hashWg sync.WaitGroup
	hashStart := time.Now()
	var hashStop time.Time

	hashWg.Add(int(ctx.numHashWorkers))

	for i := uint(0); i < ctx.numHashWorkers; i++ {
		go func() {
			defer hashWg.Done()

			for id := range ids {
				hash := hash(ctx.trees[id])
				hashes <- hashId{hash, id}
			}
		}()
	}

	go func() {
		for hashId := range hashes {
			if ctx.hashGroups[hashId.hash] == nil {
				ctx.hashGroups[hashId.hash] = make([]int, 0, 64)
			}
			ctx.hashGroups[hashId.hash] = append(ctx.hashGroups[hashId.hash], hashId.id)
		}
		done <- struct{}{}
	}()

	for i := range ctx.trees {
		ids <- i
	}
	close(ids)

	hashWg.Wait()
	close(hashes)
	<-done

	hashStop = time.Now()
	fmt.Printf("hashGroupTime: %f\n", hashStop.Sub(hashStart).Seconds())

	for hash, ids := range ctx.hashGroups {
		fmt.Printf("%d: ", hash)
		printSlice(ids)
	}
}

func hashTreesMappedMutex(ctx *context) {
	type hashId struct {hash, id int}

	ids := make(chan int, len(ctx.trees))

	lock := sync.Mutex{}

	var hashWg sync.WaitGroup
	hashStart := time.Now()
	var hashStop time.Time

	hashWg.Add(int(ctx.numHashWorkers))

	for i := uint(0); i < ctx.numHashWorkers; i++ {
		go func() {
			defer hashWg.Done()

			for id := range ids {
				hash := hash(ctx.trees[id])

				lock.Lock()
				if ctx.hashGroups[hash] == nil {
					ctx.hashGroups[hash] = make([]int, 0, 64)
				}
				ctx.hashGroups[hash] = append(ctx.hashGroups[hash], id)
				lock.Unlock()
			}
		}()
	}

	for i := range ctx.trees {
		ids <- i
	}
	close(ids)

	hashWg.Wait()

	hashStop = time.Now()
	fmt.Printf("hashGroupTime: %f\n", hashStop.Sub(hashStart).Seconds())

	for hash, ids := range ctx.hashGroups {
		fmt.Printf("%d: ", hash)
		printSlice(ids)
	}
}

func hash(bt *btree.BTree[int]) int {
	hash := 1
	bt.InOrderFunc(func(value int) {
		newValue := value + 2
		hash = (hash * newValue + newValue) % 1000
	})
	return hash
}

func printSlice(slice []int) {
	var buf bytes.Buffer

	for _, v := range slice {
		io.WriteString(&buf, fmt.Sprintf("%d ", v))
	}

	fmt.Println(buf.String())
}
