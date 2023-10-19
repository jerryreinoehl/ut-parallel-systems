package hashtable

import (
	"slices"
	"testing"
)

func TestHashTablePut(t *testing.T) {
	ht := NewHashTable[string, int]()

	ht.Put("one", 1)
	values := ht.Get("one")
	expected := []int{1}

	for i, value := range values {
		if value != expected[i] {
			t.Errorf("HashTable.Get() failed. Expected %v, got %v\n", expected, values)
		}
	}

	ht.Put("one", 11)
	ht.Put("one", 111)
	values = ht.Get("one")
	expected = []int{1, 11, 111}

	for i, value := range values {
		if value != expected[i] {
			t.Errorf("HashTable.Get() failed. Expected %v, got %v\n", expected, values)
		}
	}
}

func TestHashTableKeys(t *testing.T) {
	ht := NewHashTable[string, int]()

	ht.Put("one", 1)
	ht.Put("one", 11)
	ht.Put("one", 111)

	ht.Put("two", 2)
	ht.Put("two", 22)
	ht.Put("two", 222)

	ht.Put("three", 3)
	ht.Put("three", 33)
	ht.Put("three", 333)

	keys := ht.Keys()
	expected := []string{"one", "two", "three"}

	for _, key := range keys {
		if !slices.Contains(expected, key) {
			t.Errorf("HashTable.Keys() failed. Expected %v, got %v\n", expected, keys)
		}
	}
}
