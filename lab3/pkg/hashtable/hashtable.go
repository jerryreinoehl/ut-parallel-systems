package hashtable

import (
	"bst/pkg/cst"
	"cmp"
	"sync"
)

type hashable interface {~int | ~string}

type row[V any] struct {
	mu sync.Mutex
	data []V
}

type HashTable[K hashable, V cmp.Ordered] struct {
	mu sync.RWMutex
	data map[K]*cst.Cst[V]
}

func NewHashTable[K hashable, V cmp.Ordered] () *HashTable[K, V] {
	return &HashTable[K, V]{
		mu: sync.RWMutex{},
		data: make(map[K]*cst.Cst[V]),
	}
}

func newRow[V any]() *row[V] {
	return &row[V]{
		mu: sync.Mutex{},
		data: make([]V, 0, 32),
	}
}

func (ht *HashTable[K, V]) Put(k K, v V) {
	var row *cst.Cst[V]

	ht.mu.RLock()
	row = ht.data[k]
	ht.mu.RUnlock()

	// Check if we need to create a new row.
	if row == nil {
		ht.mu.Lock()
		// We must recheck in case this row was created before we acquired the
		// write lock.
		row = ht.data[k]
		if row == nil {
			row = cst.NewCstRef[V]()
			ht.data[k] = row
		}
		ht.mu.Unlock()
	}

	row.Insert(v)
}

func (ht *HashTable[K, V]) Get(k K) []V {
	ht.mu.RLock()
	row := ht.data[k]
	ht.mu.RUnlock()

	if row == nil {
		return nil
	}

	values := row.Items()
	return values
}

func (ht *HashTable[K, V]) Keys() []K {
	keys := make([]K, 0, 32)

	ht.mu.RLock()
	for k := range ht.data {
		keys = append(keys, k)
	}
	ht.mu.RUnlock()

	return keys
}
