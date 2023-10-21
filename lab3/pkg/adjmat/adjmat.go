// Adjacency Matrix
package adjmat

import (
	"sync"
)

type AdjMat struct {
	size int
	data [][]node
}

type node struct {
	mark  bool
	valid bool
}

type Result struct {
	i, j int
	match bool
}

func NewResult(i, j int, match bool) Result {
	return Result{i, j, match}
}

func newNode(val bool) node {
	return node{val, true}
}

func NewAdjMat(n int) AdjMat {
	mat := AdjMat{n, make([][]node, n)}
	for i := 0; i < n; i++ {
		mat.data[i] = make([]node, n)
	}

	return mat
}

func (mat *AdjMat) Set(i, j int, val bool) {
	mat.data[i][j] = newNode(val)
}

func (mat *AdjMat) Get(i, j int) bool {
	return mat.data[i][j].mark
}

func (mat *AdjMat) CmpFunc(cmp func(int, int, chan<- Result)) [][]int {
	n := mat.size

	var curGroup []int
	groups := make([][]int, 0, 8)

	// Completed rows.
	completed := make([]bool, n)
	mismatch := make([]int, 0, n)

	results := make(chan Result)
	resultsWg := sync.WaitGroup{}

	go func() {
		for result := range results {
			mat.data[result.i][result.j] = newNode(result.match)
			resultsWg.Done()
		}
	}()

	for i := 0; i < n; i++ {
		if completed[i] {
			continue
		}

		curGroup = make([]int, 0, 16)
		curGroup = append(curGroup, i)
		mismatch = mismatch[:0]

		for j := i + 1; j < n; j++ {
			if mat.data[i][j].valid {
				continue
			}
			// Schedule a comparison of i and j.
			resultsWg.Add(1)
			cmp(i, j, results)
		}
		resultsWg.Wait()

		// Scan completed row. Add matches to current group and mark their rows
		// as completed. Add mismatches to `mismatch`.
		for j := i + 1; j < n; j++ {
			if mat.data[i][j].mark {
				// j belongs to this group, no need to do anymore comparisons
				// with it.
				completed[j] = true
				curGroup = append(curGroup, j)
			} else {
				mismatch = append(mismatch, j)
			}
		}

		// Iterate through mismatches and mark mismatches with all elements in
		// the current group.
		for i := range mismatch {
			for j := range curGroup {
				if i <= j {
					continue
				}
				mat.data[i][j].mark = false
			}
		}

		groups = append(groups, curGroup)
	}

	close(results)
	return groups
}
