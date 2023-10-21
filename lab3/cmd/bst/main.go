package main

import (
	"bst/pkg/adjmat"
	"bst/pkg/btree"
	"bst/pkg/hashtable"
	"bst/pkg/queue"
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
	compareWithBuffer := flag.Bool(
		"compare-with-buffer",
		false,
		"When specified use concurrent buffer when comparing trees, otherwise use channel",
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
			if *numDataWorkers > 1 {
				hashTreesMappedSemaphore(&ctx)
			} else {
				hashTreesMappedMutex(&ctx)
			}
		} else {
			if *numDataWorkers > 1 {
				hashTreesMappedChannels(&ctx)
			} else {
				hashTreesMappedChannel(&ctx)
			}
		}
	}

	if *numCompWorkers == 1 {
		compareTreesSequential(&ctx)
	} else if *numCompWorkers > 1 {
		if *compareWithBuffer {
			compareTreesConcurrentBuffer(&ctx)
		} else {
			compareTreesConcurrent(&ctx)
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
	hashWg.Add(int(ctx.numHashWorkers))

	hashStart := time.Now()
	var hashStop time.Time

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
	hashWg.Add(int(ctx.numHashWorkers))

	hashStart := time.Now()
	var hashStop time.Time

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
		if len(ids) <= 1 {
			continue
		}

		fmt.Printf("%d: ", hash)
		printSlice(ids)
	}
}

func hashTreesMappedChannels(ctx *context) {
	type hashId struct {hash, id int}

	ids := make(chan int, len(ctx.trees))
	hashes := make(chan hashId, len(ctx.trees))
	done := make(chan struct{})

	hashGroups := hashtable.NewHashTable[int, int]()

	var hashWg sync.WaitGroup
	hashWg.Add(int(ctx.numHashWorkers))

	hashStart := time.Now()
	var hashStop time.Time

	for i := uint(0); i < ctx.numHashWorkers; i++ {
		go func() {
			defer hashWg.Done()

			for id := range ids {
				hash := hash(ctx.trees[id])
				hashes <- hashId{hash, id}
			}
		}()
	}

	for i := uint(0); i < ctx.numDataWorkers; i++ {
		go func() {
			for hashId := range hashes {
				hashGroups.Put(hashId.hash, hashId.id)
			}
			done <- struct{}{}
		}()
	}

	for i := range ctx.trees {
		ids <- i
	}
	close(ids)

	hashWg.Wait()
	close(hashes)
	<-done

	hashStop = time.Now()
	fmt.Printf("hashGroupTime: %f\n", hashStop.Sub(hashStart).Seconds())

	for _, hash := range hashGroups.Keys() {
		ids := hashGroups.Get(hash)
		ctx.hashGroups[hash] = ids
		if len(ids) <= 1 {
			continue
		}

		fmt.Printf("%d: ", hash)
		printSlice(ids)
	}
}

func hashTreesMappedMutex(ctx *context) {
	type hashId struct {hash, id int}

	ids := make(chan int, len(ctx.trees))

	lock := sync.Mutex{}

	var hashWg sync.WaitGroup
	hashWg.Add(int(ctx.numHashWorkers))

	hashStart := time.Now()
	var hashStop time.Time

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
		if len(ids) <= 1 {
			continue
		}

		fmt.Printf("%d: ", hash)
		printSlice(ids)
	}
}

func hashTreesMappedSemaphore(ctx *context) {
	type hashId struct {hash, id int}

	ids := make(chan int, len(ctx.trees))
	sema := make(chan struct{}, ctx.numDataWorkers)
	hashGroups := hashtable.NewHashTable[int, int]()

	var hashWg sync.WaitGroup
	hashStart := time.Now()
	var hashStop time.Time

	hashWg.Add(int(ctx.numHashWorkers))

	for i := uint(0); i < ctx.numHashWorkers; i++ {
		go func() {
			defer hashWg.Done()

			for id := range ids {
				hash := hash(ctx.trees[id])

				sema <- struct{}{}
				hashGroups.Put(hash, id)
				<-sema
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

	for _, hash := range hashGroups.Keys() {
		ids := hashGroups.Get(hash)
		ctx.hashGroups[hash] = ids
		if len(ids) <= 1 {
			continue
		}

		fmt.Printf("%d: ", hash)
		printSlice(ids)
	}
}

func compareTreesSequential(ctx *context) {
	allGroups := make([][]int, 0, 128)

	compStart := time.Now()
	var compStop time.Time

	for hash := range ctx.hashGroups {
		hashGroup := ctx.hashGroups[hash]
		n := len(hashGroup)

		if n <= 1 {
			continue
		}

		// Create adjacency matrix of size `n`.
		mat := adjmat.NewAdjMat(n)
		groups := mat.CmpFunc(func(i, j int, results chan<- adjmat.Result) {
			match := compareTrees(ctx.trees[hashGroup[i]], ctx.trees[hashGroup[j]])
			results <- adjmat.NewResult(i, j, match)
		})

		for _, group := range groups {
			for i, idx := range group {
				group[i] = hashGroup[idx]
			}
			allGroups = append(allGroups, group)
		}
	}

	compStop = time.Now()
	fmt.Printf("compareTreeTime: %f\n", compStop.Sub(compStart).Seconds())

	groupNum := 0
	for _, group := range allGroups {
		if len(group) <= 1 {
			continue
		}

		fmt.Printf("group %d: ", groupNum)
		printSlice(group)
		groupNum++
	}
}

func compareTreesConcurrent(ctx *context) {
	allGroups := make([][]int, 0, 128)

	type cmpItem struct {
		lhs, rhs *btree.BTree[int]
		i, j int
		result chan<- adjmat.Result
	}

	cmpQueue := make(chan cmpItem, 1024)

	compStart := time.Now()
	var compStop time.Time

	for i := uint(0); i < ctx.numCompWorkers; i++ {
		go func() {
			for cmp := range cmpQueue {
				match := compareTrees(cmp.lhs, cmp.rhs)
				cmp.result <- adjmat.NewResult(cmp.i, cmp.j, match)
			}
		}()
	}

	for hash := range ctx.hashGroups {
		hashGroup := ctx.hashGroups[hash]
		n := len(hashGroup)

		if n <= 1 {
			continue
		}

		// Create adjacency matrix of size `n`.
		mat := adjmat.NewAdjMat(n)
		groups := mat.CmpFunc(func(i, j int, results chan<- adjmat.Result) {
			cmpQueue <- cmpItem{
				lhs: ctx.trees[hashGroup[i]],
				rhs: ctx.trees[hashGroup[j]],
				i: i, j: j, result: results,
			}
		})

		for _, group := range groups {
			for i, idx := range group {
				group[i] = hashGroup[idx]
			}
			allGroups = append(allGroups, group)
		}
	}

	close(cmpQueue)

	compStop = time.Now()
	fmt.Printf("compareTreeTime: %f\n", compStop.Sub(compStart).Seconds())

	groupNum := 0
	for _, group := range allGroups {
		if len(group) <= 1 {
			continue
		}

		fmt.Printf("group %d: ", groupNum)
		printSlice(group)
		groupNum++
	}
}

func compareTreesConcurrentBuffer(ctx *context) {
	allGroups := make([][]int, 0, 128)

	type cmpItem struct {
		lhs, rhs *btree.BTree[int]
		i, j int
		result chan<- adjmat.Result
	}

	cmpQueue := queue.NewQueue[cmpItem](int(ctx.numCompWorkers))

	//cmpQueue := make(chan cmpItem, 1024)

	compStart := time.Now()
	var compStop time.Time
	compWg := sync.WaitGroup{}
	compWg.Add(int(ctx.numCompWorkers))

	for i := uint(0); i < ctx.numCompWorkers; i++ {
		go func() {
			defer compWg.Done()

			for {

				cmp, ok := cmpQueue.Dequeue()

				if !ok {
					break
				}

				match := compareTrees(cmp.lhs, cmp.rhs)
				cmp.result <- adjmat.NewResult(cmp.i, cmp.j, match)
			}
		}()
	}

	for hash := range ctx.hashGroups {
		hashGroup := ctx.hashGroups[hash]
		n := len(hashGroup)

		if n <= 1 {
			continue
		}

		// Create adjacency matrix of size `n`.
		mat := adjmat.NewAdjMat(n)
		groups := mat.CmpFunc(func(i, j int, results chan<- adjmat.Result) {
			//cmpQueue <- cmpItem{
			//	lhs: ctx.trees[hashGroup[i]],
			//	rhs: ctx.trees[hashGroup[j]],
			//	i: i, j: j, result: results,
			//}
			item := cmpItem{
				lhs: ctx.trees[hashGroup[i]],
				rhs: ctx.trees[hashGroup[j]],
				i: i, j: j, result: results,
			}
			cmpQueue.Enqueue(item)
		})

		for _, group := range groups {
			for i, idx := range group {
				group[i] = hashGroup[idx]
			}
			allGroups = append(allGroups, group)
		}
	}

	cmpQueue.Cancel()
	compWg.Wait()

	compStop = time.Now()
	fmt.Printf("compareTreeTime: %f\n", compStop.Sub(compStart).Seconds())

	groupNum := 0
	for _, group := range allGroups {
		if len(group) <= 1 {
			continue
		}

		fmt.Printf("group %d: ", groupNum)
		printSlice(group)
		groupNum++
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

func compareTrees(a, b *btree.BTree[int]) bool {
	if a.Size() != b.Size() {
		return false
	}

	aItems := a.Items()
	bItems := b.Items()

	for i := 0; i < len(aItems); i++ {
		if aItems[i] != bItems[i] {
			return false
		}
	}

	return true
}

func printSlice(slice []int) {
	var buf bytes.Buffer

	for _, v := range slice {
		io.WriteString(&buf, fmt.Sprintf("%d ", v))
	}

	fmt.Println(buf.String())
}
