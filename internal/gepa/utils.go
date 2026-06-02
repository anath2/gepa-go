package gepa

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/anath2/gepa-go/internal/program"
)

// newRNG returns a seeded random source for reproducible minibatch and Pareto sampling.
func newRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}

// sampleMinibatch picks size distinct dataset indices from [0, total).
// When size >= total, it returns every index in ascending order.
// Otherwise it uses a partial permutation and sorts the chosen indices for stable ordering.
func sampleMinibatch(rng *rand.Rand, total, size int) ([]int, error) {
	if total <= 0 {
		return nil, fmt.Errorf("sample minibatch: total must be > 0")
	}
	if size <= 0 {
		return nil, fmt.Errorf("sample minibatch: size must be > 0")
	}
	if size >= total {
		indices := make([]int, total)
		for i := range indices {
			indices[i] = i
		}
		return indices, nil
	}
	picked := rng.Perm(total)[:size]
	sort.Ints(picked)
	return picked, nil
}

// examplesAtIndices returns the train or validation rows referenced by indices.
func examplesAtIndices(examples []program.Example, indices []int) []program.Example {
	out := make([]program.Example, len(indices))
	for i, idx := range indices {
		out[i] = examples[idx]
	}
	return out
}

// scores extracts the per-example metric values from rollout results.
func scores(results []ExampleResult) []float64 {
	out := make([]float64, len(results))
	for i, result := range results {
		out[i] = result.Score
	}
	return out
}

// meanScore returns the arithmetic mean of the given scores.
func meanScore(values []float64) (float64, error) {
	if len(values) == 0 {
		return 0, fmt.Errorf("mean score: empty values")
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values)), nil
}

// strictlyImproves reports whether a proposal beats its parent on a minibatch mean score.
func strictlyImproves(parentMean, proposalMean float64) bool {
	return proposalMean > parentMean
}

// hasBudget reports whether cost additional metric calls can start without exceeding budget.
func hasBudget(metricCalls, budget, cost int) bool {
	return metricCalls+cost <= budget
}

// moduleNameAtIteration selects the module to mutate using round-robin over program.Modules.
func moduleNameAtIteration(prog program.Program, iter int) (string, error) {
	module, err := moduleAtIteration(prog, iter)
	if err != nil {
		return "", err
	}
	return module.Name, nil
}

func moduleAtIteration(prog program.Program, iter int) (program.Module, error) {
	if len(prog.Modules) == 0 {
		return program.Module{}, fmt.Errorf("module picker: program has no modules")
	}
	return prog.Modules[iter%len(prog.Modules)], nil
}

// minibatchCost returns how many metric calls a minibatch evaluation consumes.
func minibatchCost(trainLen, minibatchSize int) int {
	if minibatchSize >= trainLen {
		return trainLen
	}
	return minibatchSize
}
