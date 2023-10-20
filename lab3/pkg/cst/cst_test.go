package cst

import (
	"sync"
	"testing"
)

func TestCstInOrderFunc(t *testing.T) {
	cst := NewCst[int]()
	items := make([]int, 0)

	cst.Insert(5, 3, 7, 1, 9)

	cst.InOrderFunc(func(val int) {
		items = append(items, val)
	})
	expected := []int{1, 3, 5, 7, 9}

	for i, item := range items {
		if item != expected[i] {
			t.Errorf("cst.InOrderFunc() failed. Expected %v, got %v\n", expected, items)
		}
	}
}

func TestCstItems(t *testing.T) {
	cst := NewCst[int]()

	cst.Insert(5, 3, 7, 1, 9)
	items := cst.Items()
	expected := []int{1, 3, 5, 7, 9}

	for i, item := range items {
		if item != expected[i] {
			t.Errorf("cst.InOrderFunc() failed. Expected %v, got %v\n", expected, items)
		}
	}
}

func TestCstConcurrency(t *testing.T) {
	cst := NewCst[int]()
	insert := []int{
		23, 49, 92, 48, 45, 5, 14, 52, 68, 89, 55, 30, 7, 25, 57, 4, 70, 95,
		91, 67, 96, 13, 51, 31, 41, 27, 35, 65, 74, 24, 21, 63, 1, 83, 37, 76,
		6, 32, 10, 84, 93, 29, 90, 42, 58, 94, 39, 66, 16, 54, 80, 22, 81, 38,
		15, 2, 78, 88, 3, 61, 75, 86, 12, 56, 64, 46, 98, 85, 60, 87, 19, 97,
		8, 20, 79, 18, 34, 50, 71, 11, 99, 53, 69, 33, 47, 43, 59, 62, 40, 72,
		26, 44, 82, 77, 73, 17, 28, 9, 36,
	}

	expected := []int{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
		21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
		39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
		57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
		75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
		93, 94, 95, 96, 97, 98, 99,
	}

	threads := 16
	ch := make(chan int, 100)
	wg := sync.WaitGroup{}
	wg.Add(threads)

	for i := 0; i < threads; i++ {
		go func() {
			defer wg.Done()

			for item := range ch {
				cst.Insert(item)
			}
		}()
	}

	for _, item := range insert {
		ch <- item
	}
	close(ch)

	wg.Wait()

	items := cst.Items()
	for i, item := range items {
		if item != expected[i] {
			t.Errorf("cst.InOrderFunc() failed. Expected %v, got %v\n", expected, items)
		}
	}
}
