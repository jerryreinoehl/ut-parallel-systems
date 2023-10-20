// Concurrent search tree
package cst

import (
	"cmp"
	"sync"
)

type Cst[T cmp.Ordered] struct {
	head *node[T]
	mu sync.RWMutex
}

type node[T cmp.Ordered] struct {
	left  *node[T]
	right *node[T]
	mu    sync.RWMutex
	value T
}

func NewCst[T cmp.Ordered]() Cst[T] {
	return Cst[T]{head: nil, mu: sync.RWMutex{}}
}

func newNode[T cmp.Ordered](value T) *node[T] {
	return &node[T]{
		left: nil,
		right: nil,
		mu: sync.RWMutex{},
		value: value,
	}
}

func (cst *Cst[T]) Insert(items... T) {
	for _, item := range items {
		cst.insert(item)
	}
}

// Inserts a single item into the binary tree.
func (cst *Cst[T]) insert(item T) {
	var next *node[T] = nil
	var value T
	var left, right *node[T]
	var insertLeft bool

	// Check if head has been initialized yet.
	cst.mu.RLock()
	head := cst.head
	cst.mu.RUnlock()

	if head == nil {
		cst.mu.Lock()
		// We need to recheck head in case another thread has initialized.
		head = cst.head
		if head == nil {
			cst.head = newNode(item)
			cst.mu.Unlock()
			return
		}
		cst.mu.Unlock()
	}

	for {
		for {
			head.mu.RLock()
			value = head.value
			left = head.left
			right = head.right
			head.mu.RUnlock()

			if item < value {
				insertLeft = true
				next = left
			} else {
				insertLeft = false
				next = right
			}

			if next == nil {
				head.mu.Lock()
				break
			}

			head = next
		}

		// We need to recheck next in case another thread created this node.
		// If another thread did create this node, then unlock and continue tree
		// search.
		if insertLeft {
			next = head.left
		} else {
			next = head.right
		}

		if next != nil {
			head.mu.Unlock()
			head = next
			continue
		}

		if insertLeft {
			head.left = newNode(item)
		} else {
			head.right = newNode(item)
		}
		head.mu.Unlock()
		break
	}
}

// Traverse this cst in in-order fashion calling `fn` on each item.
func (cst *Cst[T]) InOrderFunc(fn func(T)) {
	nodes := make([]*node[T], 0, 128)
	var left, right *node[T]
	var value T

	cst.mu.RLock()
	head := cst.head
	cst.mu.RUnlock()

	for head != nil || len(nodes) > 0 {

		for head != nil {
			head.mu.RLock()
			left = head.left
			head.mu.RUnlock()

			nodes = append(nodes, head)
			head = left
		}

		head = nodes[len(nodes)-1]
		nodes = nodes[:len(nodes)-1]

		head.mu.RLock()
		right = head.right
		value = head.value
		head.mu.RUnlock()

		fn(value)
		head = right
	}
}

// Returns all the items in this btree as a slice, `[]Item`.
func (cst *Cst[T]) Items() []T {
	result := make([]T, 0)

	cst.InOrderFunc(func (item T) {
		result = append(result, item)
	})

	return result
}
