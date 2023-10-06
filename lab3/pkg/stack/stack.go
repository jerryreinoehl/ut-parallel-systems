package stack

type Stack[T any] struct {
	size int
	head *node[T]
}

type node[T any] struct {
	next  *node[T]
	value T
}

func NewStack[T any]() Stack[T] {
	return Stack[T]{size: 0, head: nil}
}

func newNode[T any](value T) *node[T] {
	return &node[T]{next: nil, value: value}
}

func (s *Stack[T]) Push(values ...T) {
	for _, v := range values {
		s.push(v)
	}
}

func (s *Stack[T]) push(value T) {
	s.size++

	if s.head == nil {
		s.head = newNode[T](value)
		return
	}

	node := newNode[T](value)
	node.next = s.head
	s.head = node
}

func (s *Stack[T]) Pop() (T, bool) {
	var value T

	if s.size == 0 {
		return value, false
	}

	s.size--
	value = s.head.value
	s.head = s.head.next

	return value, true
}

func (s *Stack[T]) Empty() bool {
	return s.size == 0
}

func (s *Stack[T]) Size() int {
	return s.size
}
