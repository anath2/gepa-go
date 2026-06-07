package gepa

import (
	"fmt"
	"math/rand"
	"sort"
)

type paretoSelector struct{}

// Draws one Pareto-frontier candidate with probability proportional to how many
// training examples they tie for the top score on (paper Alg. 2 weighted sample).
// Sorts survivor IDs before the weighted walk so a seeded RNG is deterministic.
func (paretoSelector) selectCandidate(state poolState, rng *rand.Rand) (int, error) {
	freqs, err := paretoFrequencies(state)
	if err != nil {
		return 0, err
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}

	ids := make([]int, 0, len(freqs))
	for id := range freqs {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	total := 0
	for _, id := range ids {
		total += freqs[id]
	}
	if total <= 0 {
		// All survivors have zero weight (e.g. an all-zero score pool with no
		// noise-floor columns to lift them). Fall back to a uniform draw over
		// the survivor set so selection never deadlocks.
		return ids[rng.Intn(len(ids))], nil
	}

	pick := rng.Intn(total)
	acc := 0
	for _, id := range ids {
		acc += freqs[id]
		if pick < acc {
			return id, nil
		}
	}
	// Unreachable: pick < total and the loop accumulates to total.
	return ids[len(ids)-1], nil
}

// Establishes pareto frontier from train scores
func paretoFrontier(state poolState) ([]int, error) {
	if err := validateParetoInput(state); err != nil {
		return nil, err
	}

	exampleCount := len(state.TrainScores[0])

	// Step 1: per-example winners P*[i].
	eligible := map[int]struct{}{}
	for i := 0; i < exampleCount; i++ {
		max := state.TrainScores[0][i]
		for k := 1; k < len(state.Candidates); k++ {
			if state.TrainScores[k][i] > max {
				max = state.TrainScores[k][i]
			}
		}
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

	// Step 3: prune strictly Pareto-dominated members. Strict dominance is
	// transitive, so a single pass over the eligible set is sufficient.
	survivors := candidates[:0:0]
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
	return survivors, nil
}

// Returns f[k] for each Pareto-frontier survivor: the count of training
// examples on which k ties the column max. Used as sampling weight by
// selectCandidate.
func paretoFrequencies(state poolState) (map[int]int, error) {
	survivors, err := paretoFrontier(state)
	if err != nil {
		return nil, err
	}

	exampleCount := len(state.TrainScores[0])
	freqs := make(map[int]int, len(survivors))
	for _, id := range survivors {
		freqs[id] = 0
	}

	for i := 0; i < exampleCount; i++ {
		// Find max over all candidates (matches step 1 of paretoFrontier).
		max := state.TrainScores[0][i]
		for k := 1; k < len(state.Candidates); k++ {
			if state.TrainScores[k][i] > max {
				max = state.TrainScores[k][i]
			}
		}
		for _, id := range survivors {
			if state.TrainScores[id][i] == max {
				freqs[id]++
			}
		}
	}
	return freqs, nil
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
