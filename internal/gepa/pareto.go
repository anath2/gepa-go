package gepa

import (
	"fmt"
	"math/rand"
	"sort"
)

type paretoSelector struct{}

// Draws one Pareto-frontier candidate with probability proportional to how many
// training examples they tie for the top score on.
func (paretoSelector) selectCandidate(state poolState, rng *rand.Rand) (int, error) {
	survivors, freqs, err := paretoSurvivors(state)
	if err != nil {
		return 0, err
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}

	total := 0
	for _, f := range freqs {
		total += f
	}
	if total <= 0 {
		// Reachable only with zero training examples, which leaves the frontier
		// empty; fall back to a uniform draw so selection never deadlocks.
		return survivors[rng.Intn(len(survivors))], nil
	}

	pick := rng.Intn(total)
	acc := 0
	for idx, f := range freqs {
		acc += f
		if pick < acc {
			return survivors[idx], nil
		}
	}
	// Unreachable: pick < total and the loop accumulates to total.
	return survivors[len(survivors)-1], nil
}

// Establishes pareto frontier from train scores.
func paretoFrontier(state poolState) ([]int, error) {
	survivors, _, err := paretoSurvivors(state)
	return survivors, err
}

// paretoSurvivors computes the Pareto-frontier survivor IDs together with f[k],
// the count of training examples on which each survivor ties the column max
func paretoSurvivors(state poolState) (survivors []int, freqs []int, err error) {
	if err := validateParetoInput(state); err != nil {
		return nil, nil, err
	}

	exampleCount := len(state.TrainScores[0])

	// Step 1: per-example winners P*[i]; cache each column max for the
	// frequency tally below so it is computed only once.
	maxes := make([]float64, exampleCount)
	eligible := map[int]struct{}{}
	for i := 0; i < exampleCount; i++ {
		max := state.TrainScores[0][i]
		for k := 1; k < len(state.Candidates); k++ {
			if state.TrainScores[k][i] > max {
				max = state.TrainScores[k][i]
			}
		}
		maxes[i] = max
		for k := 0; k < len(state.Candidates); k++ {
			if state.TrainScores[k][i] == max {
				eligible[k] = struct{}{}
			}
		}
	}

	// Step 2: collect eligible ids in sorted order for deterministic output.
	candidates := make([]int, 0, len(eligible))
	for id := range eligible {
		candidates = append(candidates, id)
	}
	sort.Ints(candidates)

	// Step 3: prune strictly Pareto-dominated members.
	for _, k := range candidates {
		dominated := false
		for _, j := range candidates {
			if j == k {
				continue
			}
			if dominates(state.TrainScores[j], state.TrainScores[k]) {
				dominated = true
				break
			}
		}
		if !dominated {
			survivors = append(survivors, k)
		}
	}

	// Step 4: tally f[k] using the cached column maxes.
	freqs = make([]int, len(survivors))
	for idx, id := range survivors {
		for i := 0; i < exampleCount; i++ {
			if state.TrainScores[id][i] == maxes[i] {
				freqs[idx]++
			}
		}
	}
	return survivors, freqs, nil
}

// Validate pareto frontier input.
// Shape: [C X N] where C is the number of candidates and N is the number of examples.
func validateParetoInput(state poolState) error {
	if len(state.Candidates) == 0 {
		return fmt.Errorf("pareto frontier: no candidates")
	}
	if len(state.TrainScores) != len(state.Candidates) {
		return fmt.Errorf("pareto frontier: score row count %d does not match candidate count %d", len(state.TrainScores), len(state.Candidates))
	}
	scoreLen := len(state.TrainScores[0])
	for i, candidate := range state.Candidates {
		if candidate.ID != i {
			return fmt.Errorf("pareto frontier: candidate at index %d has id %d", i, candidate.ID)
		}
		if len(state.TrainScores[i]) != scoreLen {
			return fmt.Errorf("pareto frontier: candidate %d score length %d does not match %d", i, len(state.TrainScores[i]), scoreLen)
		}
	}
	return nil
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
