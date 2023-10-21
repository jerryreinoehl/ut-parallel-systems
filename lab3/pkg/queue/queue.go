// Concurrent queue.
package queue

import (
	"sync"
)

type Queue[T any] struct {
	data        []T
	consumeCond *sync.Cond // Signals that there are items available in queue.
	produceCond *sync.Cond // Signals that items may be added to queue.
	capacity    int
	cancel      bool
	waiting     int
}

func NewQueue[T any](capacity int) Queue[T] {
	mu := sync.Mutex{}

	return Queue[T]{
		data: make([]T, 0),
		consumeCond: sync.NewCond(&mu),
		produceCond: sync.NewCond(&mu),
		capacity: capacity,
		cancel: false,
		waiting: 0,
	}
}

func (q *Queue[T]) Enqueue(t T) {
	q.produceCond.L.Lock()
	for q.Size() == q.capacity {
		q.produceCond.Wait()
		if q.cancel {
			q.produceCond.L.Unlock()
			return
		}
	}
	q.data = append(q.data, t)
	q.consumeCond.Signal()
	q.produceCond.L.Unlock()
}

func (q *Queue[T]) Dequeue() (T, bool) {
	var t T

	q.consumeCond.L.Lock()

	for q.Size() == 0 {
		q.waiting++
		q.consumeCond.Wait()
		q.waiting--
		if q.cancel {
			q.produceCond.Signal()
			q.consumeCond.L.Unlock()
			return t, false
		}
	}

	t = q.data[0]
	q.data = q.data[1:]

	q.produceCond.Signal()
	q.consumeCond.L.Unlock()

	return t, true
}

func (q *Queue[T]) Size() int {
	return len(q.data)
}

func (q *Queue[T]) Cancel() {
	q.consumeCond.L.Lock()
	q.cancel = true
	q.consumeCond.L.Unlock()

	for q.waiting > 0 {
		q.consumeCond.L.Lock()
		q.consumeCond.Broadcast()
		q.produceCond.Broadcast()
		q.produceCond.Wait()
		q.consumeCond.L.Unlock()
	}
}
