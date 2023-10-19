package hashtable

import (
	"sync"
)

type hashable interface {~int | ~string}

type row[V any] struct {
	mu sync.Mutex
	data []V
}

type HashTable[K hashable, V any] struct {
	mu sync.RWMutex
	data map[K]*row[V]
}

func NewHashTable[K hashable, V any] () *HashTable[K, V] {
	return &HashTable[K, V]{
		mu: sync.RWMutex{},
		data: make(map[K]*row[V]),
	}
}

func newRow[V any]() *row[V] {
	return &row[V]{
		mu: sync.Mutex{},
		data: make([]V, 0, 32),
	}
}

func (ht *HashTable[K, V]) Put(k K, v V) {
	var row *row[V]

	ht.mu.RLock()
	// Check if we need to create a new row.
	if ht.data[k] == nil {
		ht.mu.RUnlock()
		ht.mu.Lock()
		// We must recheck in case this row was created before we acquired the
		// write lock.
		if ht.data[k] == nil {
			ht.data[k] = newRow[V]()
		}
		row = ht.data[k]
		ht.mu.Unlock()
	} else {
		row = ht.data[k]
		ht.mu.RUnlock()
	}

	// Acquire lock for row.
	row.mu.Lock()
	row.data = append(row.data, v)
	row.mu.Unlock()
}

func (ht *HashTable[K, V]) Get(k K) []V {
	ht.mu.RLock()
	if ht.data[k] == nil {
		return nil
	}
	ht.mu.RUnlock()

	row := ht.data[k]
	row.mu.Lock()
	v := row.data
	row.mu.Unlock()

	return v
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
