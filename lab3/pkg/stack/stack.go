package stack

type any = interface{}

type Stack struct {
	size int
	head *node
}

type node struct {
	next  *node
	value any
}

func NewStack() Stack {
	return Stack{size: 0, head: nil}
}

func newNode(value any) *node {
	return &node{next: nil, value: value}
}

func (s *Stack) Push(values ...any) {
	for _, v := range values {
		s.push(v)
	}
}

func (s *Stack) push(value any) {
	s.size++

	if s.head == nil {
		s.head = newNode(value)
		return
	}

	node := newNode(value)
	node.next = s.head
	s.head = node
}

func (s *Stack) Pop() any {
	if s.size == 0 {
		return nil
	}

	s.size--
	value := s.head.value
	s.head = s.head.next
	return value
}

func (s *Stack) Empty() bool {
	return s.size == 0
}

func (s *Stack) Size() int {
	return s.size
}
