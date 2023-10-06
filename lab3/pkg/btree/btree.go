package btree

import (
	"bst/pkg/stack"
	"cmp"
)

type BTree[T cmp.Ordered] struct {
	head *node[T]
	size int
}

type node[T cmp.Ordered] struct {
	left  *node[T]
	right *node[T]
	value T
}

func NewBTree[T cmp.Ordered]() BTree[T] {
	return BTree[T]{head: nil, size: 0}
}

func newNode[T cmp.Ordered](value T) *node[T] {
	return &node[T]{left: nil, right: nil, value: value}
}

func (b *BTree[T]) Size() int {
	return b.size
}

// Inserts items into the binary tree.
func (b *BTree[T]) Insert(items ...T) {
	for _, item := range items {
		b.insert(item)
	}
}

// Inserts a single item into the binary tree.
func (b *BTree[T]) insert(item T) {
	var next *node[T] = nil

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
func (b *BTree[T]) InOrderFunc(fn func(T)) {
	nodes := stack.NewStack[*node[T]]()
	ptr := b.head

	for ptr != nil || !nodes.Empty() {
		for ptr != nil {
			nodes.Push(ptr)
			ptr = ptr.left
		}

		ptr, _ = nodes.Pop()
		fn(ptr.value)
		ptr = ptr.right
	}
}

// Returns all the items in this btree as a slice, `[]Item`.
func (b *BTree[T]) Items() []T {
	result := make([]T, 0, b.size)

	b.InOrderFunc(func (item T) {
		result = append(result, item)
	})

	return result
}
