package gepa

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestParetoFrontierSingleCandidate(t *testing.T) {
	state := stateWithScores([][]float64{{1, 0, 1}})

	got, err := paretoFrontier(state)
	if err != nil {
		t.Fatalf("paretoFrontier() unexpected error: %v", err)
	}
	want := []int{0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paretoFrontier() = %#v, want %#v", got, want)
	}
}

func TestParetoFrontierExcludesDominatedCandidates(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 1, 1},
		{1, 0, 1},
		{0, 1, 1},
	})

	got, err := paretoFrontier(state)
	if err != nil {
		t.Fatalf("paretoFrontier() unexpected error: %v", err)
	}
	want := []int{0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paretoFrontier() = %#v, want %#v", got, want)
	}
}

func TestParetoFrontierKeepsTradeoffsAndTies(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0},
		{0, 1},
		{1, 0},
	})

	got, err := paretoFrontier(state)
	if err != nil {
		t.Fatalf("paretoFrontier() unexpected error: %v", err)
	}
	want := []int{0, 1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paretoFrontier() = %#v, want %#v", got, want)
	}
}

func TestParetoFrontierRejectsMissingScoreRows(t *testing.T) {
	state := poolState{
		Candidates: []candidateRecord{
			{ID: 0, ParentIDs: []int{}, ProposalKind: proposalSeed},
			{ID: 1, ParentIDs: []int{0}, ProposalKind: proposalReflection},
		},
		TrainScores: [][]float64{{1}},
	}

	if _, err := paretoFrontier(state); err == nil {
		t.Fatal("paretoFrontier() error = nil, want missing score row error")
	}
}

func TestParetoFrontierRejectsIncompleteScoreVectors(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0},
		{0},
	})

	if _, err := paretoFrontier(state); err == nil {
		t.Fatal("paretoFrontier() error = nil, want incomplete score vector error")
	}
}

// TestParetoFrontierContinuousScoresExcludesNonWinners verifies that, under
// continuous scores, the union-of-per-example-winners construction excludes
// candidates that never lead on any example even though they are not strictly
// Pareto-dominated by another (paper Alg. 2 step 1-3).
func TestParetoFrontierContinuousScoresExcludesNonWinners(t *testing.T) {
	state := stateWithScores([][]float64{
		{0.5, 0.5}, // A: never strictly tops either example
		{0.6, 0.4}, // B: leads example 0
		{0.4, 0.6}, // C: leads example 1
	})

	got, err := paretoFrontier(state)
	if err != nil {
		t.Fatalf("paretoFrontier() unexpected error: %v", err)
	}
	want := []int{1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paretoFrontier() = %#v, want %#v", got, want)
	}
}

// TestParetoFrontierBinaryEquivalence ensures the new survivor set matches the
// classical Pareto frontier under binary scoring, which is what every existing
// call site (notably merge.go) assumes.
func TestParetoFrontierBinaryEquivalence(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 1, 0, 0}, // A
		{1, 0, 1, 0}, // B
		{0, 1, 0, 1}, // C
	})

	got, err := paretoFrontier(state)
	if err != nil {
		t.Fatalf("paretoFrontier() unexpected error: %v", err)
	}
	// None of A, B, C strictly Pareto-dominates the others. All survive.
	want := []int{0, 1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paretoFrontier() = %#v, want %#v", got, want)
	}
}

// TestParetoFrequenciesHandComputed asserts the f[k] values directly for a
// fixture small enough to verify by inspection. Each candidate counts the
// number of examples on which it still ties for the max after the dominance
// prune.
func TestParetoFrequenciesHandComputed(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 1, 1, 0}, // A leads examples 0,1,2
		{1, 0, 0, 1}, // B leads examples 0,3
		{0, 1, 0, 1}, // C leads examples 1,3
	})

	freqs, err := paretoFreqMap(state)
	if err != nil {
		t.Fatalf("paretoSurvivors() unexpected error: %v", err)
	}
	want := map[int]int{0: 3, 1: 2, 2: 2}
	if !reflect.DeepEqual(freqs, want) {
		t.Fatalf("paretoSurvivors() freqs = %#v, want %#v", freqs, want)
	}
}

// paretoFreqMap adapts the index-aligned paretoSurvivors output into a
// {candidateID: f[k]} map for assertions on the frequency tally.
func paretoFreqMap(state poolState) (map[int]int, error) {
	survivors, freqs, err := paretoSurvivors(state)
	if err != nil {
		return nil, err
	}
	out := make(map[int]int, len(survivors))
	for i, id := range survivors {
		out[id] = freqs[i]
	}
	return out, nil
}

// TestParetoFrequenciesNobodySolvedColumn checks the "noise-floor" column case:
// a column where every candidate ties at 0 should add +1 to every survivor's
// frequency because every survivor is in P*[i] for that example.
func TestParetoFrequenciesNobodySolvedColumn(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0, 0}, // A leads example 0; tied at 0 on example 2
		{0, 1, 0}, // B leads example 1; tied at 0 on example 2
	})

	freqs, err := paretoFreqMap(state)
	if err != nil {
		t.Fatalf("paretoSurvivors() unexpected error: %v", err)
	}
	want := map[int]int{0: 2, 1: 2}
	if !reflect.DeepEqual(freqs, want) {
		t.Fatalf("paretoSurvivors() freqs = %#v, want %#v", freqs, want)
	}
}

// TestParetoSelectorWeightedSamplingDistribution verifies that selectCandidate
// draws proportionally to f[k] on a fixture with f[A]=3, f[B]=2, f[C]=2.
func TestParetoSelectorWeightedSamplingDistribution(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 1, 1, 0}, // A
		{1, 0, 0, 1}, // B
		{0, 1, 0, 1}, // C
	})
	selector := paretoSelector{}
	rng := rand.New(rand.NewSource(1))

	const trials = 100_000
	counts := map[int]int{0: 0, 1: 0, 2: 0}
	for range trials {
		got, err := selector.selectCandidate(state, rng)
		if err != nil {
			t.Fatalf("selectCandidate() unexpected error: %v", err)
		}
		counts[got]++
	}

	want := map[int]float64{0: 3.0 / 7.0, 1: 2.0 / 7.0, 2: 2.0 / 7.0}
	const tol = 0.01
	for id, expected := range want {
		got := float64(counts[id]) / float64(trials)
		if math.Abs(got-expected) > tol {
			t.Errorf("candidate %d empirical=%.4f, want %.4f +/- %.4f", id, got, expected, tol)
		}
	}
}

func TestParetoSelectorSingleCandidate(t *testing.T) {
	state := stateWithScores([][]float64{{1, 0, 1}})
	selector := paretoSelector{}
	rng := rand.New(rand.NewSource(42))

	for range 5 {
		got, err := selector.selectCandidate(state, rng)
		if err != nil {
			t.Fatalf("selectCandidate() unexpected error: %v", err)
		}
		if got != 0 {
			t.Fatalf("selectCandidate() = %d, want 0", got)
		}
	}
}

// TestParetoFrontierAllRowsIdentical confirms that when every candidate has
// the same score row, none is strictly dominated and all survive. Sampling
// should then be uniform because each candidate has identical frequency.
func TestParetoFrontierAllRowsIdentical(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0, 1},
		{1, 0, 1},
		{1, 0, 1},
	})

	got, err := paretoFrontier(state)
	if err != nil {
		t.Fatalf("paretoFrontier() unexpected error: %v", err)
	}
	want := []int{0, 1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paretoFrontier() = %#v, want %#v", got, want)
	}

	freqs, err := paretoFreqMap(state)
	if err != nil {
		t.Fatalf("paretoSurvivors() unexpected error: %v", err)
	}
	// Every column has all three at the max, so each candidate gets +1 per column.
	// 3 columns * 1 each = 3.
	wantFreqs := map[int]int{0: 3, 1: 3, 2: 3}
	if !reflect.DeepEqual(freqs, wantFreqs) {
		t.Fatalf("paretoSurvivors() freqs = %#v, want %#v", freqs, wantFreqs)
	}

	selector := paretoSelector{}
	rng := rand.New(rand.NewSource(7))
	const trials = 60_000
	counts := map[int]int{0: 0, 1: 0, 2: 0}
	for range trials {
		got, err := selector.selectCandidate(state, rng)
		if err != nil {
			t.Fatalf("selectCandidate() unexpected error: %v", err)
		}
		counts[got]++
	}
	const tol = 0.01
	for id := 0; id < 3; id++ {
		freq := float64(counts[id]) / float64(trials)
		if math.Abs(freq-1.0/3.0) > tol {
			t.Errorf("candidate %d empirical=%.4f, want %.4f +/- %.4f", id, freq, 1.0/3.0, tol)
		}
	}
}

// TestParetoFrontierAllZeros covers the degenerate case where no candidate has
// scored above zero anywhere. Nobody dominates anybody, so all survive and
// sampling is uniform.
func TestParetoFrontierAllZeros(t *testing.T) {
	state := stateWithScores([][]float64{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	})

	got, err := paretoFrontier(state)
	if err != nil {
		t.Fatalf("paretoFrontier() unexpected error: %v", err)
	}
	want := []int{0, 1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paretoFrontier() = %#v, want %#v", got, want)
	}

	selector := paretoSelector{}
	rng := rand.New(rand.NewSource(11))
	const trials = 60_000
	counts := map[int]int{0: 0, 1: 0, 2: 0}
	for range trials {
		got, err := selector.selectCandidate(state, rng)
		if err != nil {
			t.Fatalf("selectCandidate() unexpected error: %v", err)
		}
		counts[got]++
	}
	const tol = 0.01
	for id := 0; id < 3; id++ {
		freq := float64(counts[id]) / float64(trials)
		if math.Abs(freq-1.0/3.0) > tol {
			t.Errorf("candidate %d empirical=%.4f, want %.4f +/- %.4f", id, freq, 1.0/3.0, tol)
		}
	}
}

// TestParetoSelectorSeededDeterminism documents that two RNGs seeded
// identically produce identical sequences under the weighted draw.
func TestParetoSelectorSeededDeterminism(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0},
		{0, 0},
		{0, 1},
	})
	selector := paretoSelector{}
	left := rand.New(rand.NewSource(42))
	right := rand.New(rand.NewSource(42))

	var leftSeq, rightSeq []int
	for range 12 {
		got, err := selector.selectCandidate(state, left)
		if err != nil {
			t.Fatalf("selectCandidate() unexpected error: %v", err)
		}
		if got == 1 {
			t.Fatalf("selectCandidate() selected non-winner candidate %d", got)
		}
		leftSeq = append(leftSeq, got)

		again, err := selector.selectCandidate(state, right)
		if err != nil {
			t.Fatalf("selectCandidate() second rng unexpected error: %v", err)
		}
		rightSeq = append(rightSeq, again)
	}

	if !reflect.DeepEqual(leftSeq, rightSeq) {
		t.Fatalf("seeded selection mismatch: left=%v right=%v", leftSeq, rightSeq)
	}
}

func stateWithScores(scores [][]float64) poolState {
	candidates := make([]candidateRecord, len(scores))
	for i := range scores {
		candidates[i] = candidateRecord{
			ID:           i,
			ParentIDs:    []int{},
			ProposalKind: proposalSeed,
		}
	}
	return poolState{Candidates: candidates, TrainScores: scores}
}
