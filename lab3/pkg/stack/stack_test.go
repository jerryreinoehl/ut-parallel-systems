package stack

import "testing"

func TestStackSize(t *testing.T) {
	s := NewStack[int]()
	expectedSize := 0

	if s.Size() != expectedSize {
		t.Errorf("stack should have size %d but has size %d", expectedSize, s.Size())
	}

	s.Push(0)
	expectedSize = 1
	if s.Size() != expectedSize {
		t.Errorf("stack should have size %d but has size %d", expectedSize, s.Size())
	}

	s.Push(10, 20, 30, 40, 50)
	expectedSize = 6
	if s.Size() != expectedSize {
		t.Errorf("stack should have size %d but has size %d", expectedSize, s.Size())
	}

	_, ok := s.Pop()
	expectedSize = 5

	if !ok {
		t.Errorf("Pop() returned false for ok, expected true")
	}

	if s.Size() != expectedSize {
		t.Errorf("stack should have size %d but has size %d", expectedSize, s.Size())
	}

	_, ok = s.Pop()
	_, ok = s.Pop()
	_, ok = s.Pop()
	expectedSize = 2

	if !ok {
		t.Errorf("Pop() returned false for ok, expected true")
	}

	if s.Size() != expectedSize {
		t.Errorf("stack should have size %d but has size %d", expectedSize, s.Size())
	}

	_, ok = s.Pop()
	_, ok = s.Pop()
	expectedSize = 0

	if ok {
		t.Errorf("Pop() returned true on empty stack, expected false")
	}

	if s.Size() != expectedSize {
		t.Errorf("stack should have size %d but has size %d", expectedSize, s.Size())
	}

	_, ok = s.Pop()
	_, ok = s.Pop()

	if ok {
		t.Errorf("Pop() returned true on empty stack, expected false")
	}

	if s.Size() != expectedSize {
		t.Errorf("stack should have size %d but has size %d", expectedSize, s.Size())
	}
}

func TestStackOrder(t *testing.T) {
	s := NewStack[int]()

	s.Push(1)
	value, _ := s.Pop()
	expected := 1
	if value != expected {
		t.Errorf("Expected Pop() to return value %d, but got %d", expected, value)
	}

	s.Push(10, 20, 30, 40, 50)
	value, _ = s.Pop()
	expected = 50
	if value != expected {
		t.Errorf("Expected Pop() to return value %d, but got %d", expected, value)
	}

	value, _ = s.Pop()
	expected = 40
	if value != expected {
		t.Errorf("Expected Pop() to return value %d, but got %d", expected, value)
	}

	value, _ = s.Pop()
	expected = 30
	if value != expected {
		t.Errorf("Expected Pop() to return value %d, but got %d", expected, value)
	}

	value, _ = s.Pop()
	expected = 20
	if value != expected {
		t.Errorf("Expected Pop() to return value %d, but got %d", expected, value)
	}

	value, _ = s.Pop()
	expected = 10
	if value != expected {
		t.Errorf("Expected Pop() to return value %d, but got %d", expected, value)
	}
}

func TestStackEmpty(t *testing.T) {
	s := NewStack[int]()

	if !s.Empty() {
		t.Errorf("Expected Empty() to return true, but got false")
	}

	s.Push(1)
	if s.Empty() {
		t.Errorf("Expected Empty() to return false, but got true")
	}

	s.Push(10, 20, 30, 40, 50)
	if s.Empty() {
		t.Errorf("Expected Empty() to return false, but got true")
	}

	s.Pop()
	if s.Empty() {
		t.Errorf("Expected Empty() to return false, but got true")
	}

	s.Pop()
	s.Pop()
	s.Pop()
	s.Pop()
	s.Pop()
	if !s.Empty() {
		t.Errorf("Expected Empty() to return false, but got true")
	}
}
