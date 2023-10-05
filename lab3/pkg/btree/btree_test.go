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
