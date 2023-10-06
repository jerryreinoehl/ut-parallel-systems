package main

import (
	"fmt"
	"bst/pkg/btree"
	"bufio"
	"os"
	"log"
	"strings"
	"strconv"
)

func main() {
	bt := btree.NewBTree()
	bt.Insert(10, 23, 11, 5)
	bt.Insert(10)

	fmt.Println(bt.Size())
	fmt.Printf("%v\n", bt.Items())

	trees := loadTrees("data/simple.txt")
	for _, tree := range trees {
		fmt.Println(tree.Items())
	}
}

func loadTrees(file string) []*btree.BTree {
	trees := make([]*btree.BTree, 0)

	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}

	defer f.Close()

	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		tree := btree.NewBTree()
		trees = append(trees, &tree)

		line := scanner.Text()
		fmt.Println(line)

		for _, item := range strings.Split(line, " ") {
			parsed, err := strconv.Atoi(item)
			if err != nil {
				log.Fatal(err)
			}
			tree.Insert(btree.Item(parsed))
		}
	}

	return trees
}
