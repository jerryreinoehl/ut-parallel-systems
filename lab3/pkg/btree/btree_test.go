package btree

import (
	"testing"
)

func TestEmptyBTreeSize(t *testing.T) {
	bt := NewBTree()
	expectedSize := 0

	if bt.Size != expectedSize {
		t.Errorf("btree should have size %d but has size %d", expectedSize, bt.Size)
	}

	bt.Insert(1)
	expectedSize = 1

	if bt.Size != expectedSize {
		t.Errorf("btree should have size %d but has size %d", expectedSize, bt.Size)
	}

	bt.Insert(10, 20, 30, 40, 50)
	expectedSize = 6

	if bt.Size != expectedSize {
		t.Errorf("btree should have size %d but has size %d", expectedSize, bt.Size)
	}
}

func TestInOrderFunc(t *testing.T) {
	bt := NewBTree()
	values := make([]int, 0, 10)

	bt.Insert(30, 15, 10, 25, 40, 35, 50, 20, 100, 17)
	bt.InOrderFunc(func (i Item) {
		values = append(values, int(i))
	})

	expected := []int{10, 15, 17, 20, 25, 30, 35, 40, 50, 100}

	for i, value := range values {
		if value != expected[i] {
			t.Errorf("btree.InOrderFunc() failed. Expected %v, got %v\n", expected, values)
		}
	}
}
