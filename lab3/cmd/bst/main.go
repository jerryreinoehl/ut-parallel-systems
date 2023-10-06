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
)

func main() {
	numHashWorkers := flag.Uint("hash-workers", 1, "Number of hash workers")
	numDataWorkers := flag.Uint("data-workers", 0, "Number of data workers")
	numCompWorkers := flag.Uint("comp-workers", 0, "Number of comparison workers")
	input := flag.String("input", "", "Input file path")
	flag.Parse()

	if *input == "" {
		log.Fatal("Must specify input")
	}

	fmt.Printf("numHashWorkers = %d\n", *numHashWorkers)
	fmt.Printf("numDataWorkers = %d\n", *numDataWorkers)
	fmt.Printf("numCompWorkers = %d\n", *numCompWorkers)

	bt := btree.NewBTree[int]()
	bt.Insert(10, 23, 11, 5)
	bt.Insert(10)

	fmt.Println(bt.Size())
	fmt.Printf("%v\n", bt.Items())

	trees := loadTrees(*input)
	for _, tree := range trees {
		fmt.Println(tree.Items())
		fmt.Printf("hash: %d\n", hash(tree))
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

func hash(bt *btree.BTree[int]) int {
	hash := 1
	bt.InOrderFunc(func(value int) {
		newValue := value + 2
		hash = (hash * newValue + newValue) % 1000
	})
	return hash
}
