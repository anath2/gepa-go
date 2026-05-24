package gepa

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestParetoFrontierSingleCandidate(t *testing.T) {
	state := stateWithScores([][]float64{{1, 0, 1}})

	got, err := ParetoFrontier(state)
	if err != nil {
		t.Fatalf("ParetoFrontier() unexpected error: %v", err)
	}
	want := []int{0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParetoFrontier() = %#v, want %#v", got, want)
	}
}

func TestParetoFrontierExcludesDominatedCandidates(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 1, 1},
		{1, 0, 1},
		{0, 1, 1},
	})

	got, err := ParetoFrontier(state)
	if err != nil {
		t.Fatalf("ParetoFrontier() unexpected error: %v", err)
	}
	want := []int{0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParetoFrontier() = %#v, want %#v", got, want)
	}
}

func TestParetoFrontierKeepsTradeoffsAndTies(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0},
		{0, 1},
		{1, 0},
	})

	got, err := ParetoFrontier(state)
	if err != nil {
		t.Fatalf("ParetoFrontier() unexpected error: %v", err)
	}
	want := []int{0, 1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParetoFrontier() = %#v, want %#v", got, want)
	}
}

func TestParetoFrontierRejectsMissingScoreRows(t *testing.T) {
	state := State{
		Candidates: []CandidateRecord{
			{ID: 0, ParentIDs: []int{}, ProposalKind: ProposalSeed},
			{ID: 1, ParentIDs: []int{0}, ProposalKind: ProposalReflection},
		},
		TrainScores: [][]float64{{1}},
	}

	if _, err := ParetoFrontier(state); err == nil {
		t.Fatal("ParetoFrontier() error = nil, want missing score row error")
	}
}

func TestParetoFrontierRejectsIncompleteScoreVectors(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0},
		{0},
	})

	if _, err := ParetoFrontier(state); err == nil {
		t.Fatal("ParetoFrontier() error = nil, want incomplete score vector error")
	}
}

func TestParetoSelectorSamplesOnlyFromFrontierDeterministically(t *testing.T) {
	state := stateWithScores([][]float64{
		{1, 0},
		{0, 0},
		{0, 1},
	})
	selector := ParetoSelector{}
	left := rand.New(rand.NewSource(42))
	right := rand.New(rand.NewSource(42))

	var leftSeq, rightSeq []int
	for range 12 {
		got, err := selector.SelectCandidate(state, left)
		if err != nil {
			t.Fatalf("SelectCandidate() unexpected error: %v", err)
		}
		if got == 1 {
			t.Fatalf("SelectCandidate() selected dominated candidate %d", got)
		}
		leftSeq = append(leftSeq, got)

		again, err := selector.SelectCandidate(state, right)
		if err != nil {
			t.Fatalf("SelectCandidate() second rng unexpected error: %v", err)
		}
		rightSeq = append(rightSeq, again)
	}

	if !reflect.DeepEqual(leftSeq, rightSeq) {
		t.Fatalf("seeded selection mismatch: left=%v right=%v", leftSeq, rightSeq)
	}
}

func stateWithScores(scores [][]float64) State {
	candidates := make([]CandidateRecord, len(scores))
	for i := range scores {
		candidates[i] = CandidateRecord{
			ID:           i,
			ParentIDs:    []int{},
			ProposalKind: ProposalSeed,
		}
	}
	return State{Candidates: candidates, TrainScores: scores}
}
