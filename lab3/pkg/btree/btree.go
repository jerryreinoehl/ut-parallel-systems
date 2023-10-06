package btree

import (
	"bst/pkg/stack"
)

// We should really use generics but are limited to go 1.12 here so just use
// `int`.
type Item int

type BTree struct {
	head *node
	size int
}

type node struct {
	left  *node
	right *node
	value Item
}

func NewBTree() BTree {
	return BTree{head: nil, size: 0}
}

func newNode(value Item) *node {
	return &node{left: nil, right: nil, value: value}
}

func (b *BTree) Size() int {
	return b.size
}

// Inserts items into the binary tree.
func (b *BTree) Insert(items ...Item) {
	for _, item := range items {
		b.insert(item)
	}
}

// Inserts a single item into the binary tree.
func (b *BTree) insert(item Item) {
	var next *node = nil

	b.size++

	if b.head == nil {
		b.head = newNode(item)
		return
	}

	p := b.head

	for {
		if item < p.value {
			next = p.left
		} else {
			next = p.right
		}

		if next == nil {
			break
		}

		p = next
	}

	if item < p.value {
		p.left = newNode(item)
	} else {
		p.right = newNode(item)
	}
}

// Traverse this btree in in-order fashion calling `fn` on each item.
func (b *BTree) InOrderFunc(fn func(Item)) {
	nodes := stack.NewStack()
	ptr := b.head

	for ptr != nil || !nodes.Empty() {
		for ptr != nil {
			nodes.Push(ptr)
			ptr = ptr.left
		}

		ptr = nodes.Pop().(*node)
		fn(ptr.value)
		ptr = ptr.right
	}
}

// Returns all the items in this btree as a slice, `[]Item`.
func (b *BTree) Items() []Item {
	result := make([]Item, 0, b.size)

	b.InOrderFunc(func (item Item) {
		result = append(result, item)
	})

	return result
}
