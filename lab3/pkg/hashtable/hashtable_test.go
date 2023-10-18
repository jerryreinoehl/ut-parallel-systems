package hashtable

import (
	"testing"
)

func TestHashTable(t *testing.T) {
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
