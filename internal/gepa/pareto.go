package gepa

import (
	"fmt"
	"math/rand"
)

type paretoSelector struct{}

func (paretoSelector) selectCandidate(state poolState, rng *rand.Rand) (int, error) {
	frontier, err := paretoFrontier(state)
	if err != nil {
		return 0, err
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}
	return frontier[rng.Intn(len(frontier))], nil
}

func paretoFrontier(state poolState) ([]int, error) {
	if len(state.Candidates) == 0 {
		return nil, fmt.Errorf("pareto frontier: no candidates")
	}
	if len(state.TrainScores) != len(state.Candidates) {
		return nil, fmt.Errorf("pareto frontier: score row count %d does not match candidate count %d", len(state.TrainScores), len(state.Candidates))
	}

	scoreLen := len(state.TrainScores[0])
	for i, candidate := range state.Candidates {
		if candidate.ID != i {
			return nil, fmt.Errorf("pareto frontier: candidate at index %d has id %d", i, candidate.ID)
		}
		if len(state.TrainScores[i]) != scoreLen {
			return nil, fmt.Errorf("pareto frontier: candidate %d score length %d does not match %d", i, len(state.TrainScores[i]), scoreLen)
		}
	}

	var frontier []int
	for i := range state.Candidates {
		dominated := false
		for j := range state.Candidates {
			if i == j {
				continue
			}
			if dominates(state.TrainScores[j], state.TrainScores[i]) {
				dominated = true
				break
			}
		}
		if !dominated {
			frontier = append(frontier, i)
		}
	}
	return frontier, nil
}

func dominates(a, b []float64) bool {
	strictlyBetter := false
	for i := range a {
		if a[i] < b[i] {
			return false
		}
		if a[i] > b[i] {
			strictlyBetter = true
		}
	}
	return strictlyBetter
}
