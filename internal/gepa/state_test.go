package gepa

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anath2/gepa-go/internal/program"
)

func TestSeedCandidateFromProgram(t *testing.T) {
	prog := program.Program{
		Modules: []program.Module{
			{Name: "segmenter", Prompt: "segment text"},
			{Name: "ner", Prompt: "label entities"},
		},
	}

	got := SeedCandidate(prog)
	want := Candidate{
		"segmenter": "segment text",
		"ner":       "label entities",
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("SeedCandidate() = %#v, want %#v", got, want)
	}
}

func TestSeedRecordHasMergeReadyLineage(t *testing.T) {
	prog := program.Program{
		Modules: []program.Module{
			{Name: "answer", Prompt: "answer the question"},
		},
	}

	record := NewSeedRecord(prog)

	if record.ID != 0 {
		t.Fatalf("seed ID = %d, want 0", record.ID)
	}
	if record.ProposalKind != ProposalSeed {
		t.Fatalf("seed proposal kind = %q, want %q", record.ProposalKind, ProposalSeed)
	}
	if record.ParentIDs == nil {
		t.Fatal("seed ParentIDs is nil, want empty slice for stable JSON lineage shape")
	}
	if len(record.ParentIDs) != 0 {
		t.Fatalf("seed ParentIDs = %#v, want empty", record.ParentIDs)
	}
	if record.MutatedModule != "" {
		t.Fatalf("seed MutatedModule = %q, want empty", record.MutatedModule)
	}
	if got := record.Prompts["answer"]; got != "answer the question" {
		t.Fatalf("seed prompt = %q, want %q", got, "answer the question")
	}
}

func TestCandidateRecordSupportsReflectionAndFutureMergeLineage(t *testing.T) {
	reflection := CandidateRecord{
		ID:            1,
		ParentIDs:     []int{0},
		ProposalKind:  ProposalReflection,
		MutatedModule: "ner",
		CreatedAtIter: 3,
		Prompts:       Candidate{"ner": "improved prompt"},
	}
	if !reflect.DeepEqual(reflection.ParentIDs, []int{0}) {
		t.Fatalf("reflection ParentIDs = %#v, want [0]", reflection.ParentIDs)
	}
	if reflection.ProposalKind != ProposalReflection {
		t.Fatalf("reflection kind = %q, want %q", reflection.ProposalKind, ProposalReflection)
	}

	merge := CandidateRecord{
		ID:            2,
		ParentIDs:     []int{0, 1},
		ProposalKind:  ProposalMerge,
		CreatedAtIter: 4,
		Prompts:       Candidate{"ner": "merged prompt"},
	}
	if !reflect.DeepEqual(merge.ParentIDs, []int{0, 1}) {
		t.Fatalf("merge ParentIDs = %#v, want [0 1]", merge.ParentIDs)
	}
	if merge.ProposalKind != ProposalMerge {
		t.Fatalf("merge kind = %q, want %q", merge.ProposalKind, ProposalMerge)
	}
}

func testMutationProgram() program.Program {
	return program.Program{
		Modules: []program.Module{{Name: "answer", Prompt: "seed prompt"}},
	}
}

func TestNewState_HasSeedOnly(t *testing.T) {
	state := NewState(testMutationProgram())
	if len(state.Candidates) != 1 {
		t.Fatalf("Candidates len = %d, want 1", len(state.Candidates))
	}
	if state.Candidates[0].ID != 0 || state.Candidates[0].ProposalKind != ProposalSeed {
		t.Fatalf("seed record = %#v", state.Candidates[0])
	}
	if len(state.TrainScores) != 1 || state.TrainScores[0] != nil {
		t.Fatalf("TrainScores = %#v, want one nil row", state.TrainScores)
	}
	if state.BestCandidate != 0 {
		t.Fatalf("BestCandidate = %d, want 0", state.BestCandidate)
	}
}

func TestSetSeedTrainScores_UpdatesRowAndBest(t *testing.T) {
	state := NewState(testMutationProgram())
	trainLen := 3
	scores := []float64{0.5, 0.5, 0.5}
	if err := SetSeedTrainScores(&state, trainLen, scores); err != nil {
		t.Fatalf("SetSeedTrainScores() unexpected error: %v", err)
	}
	if len(state.TrainScores[0]) != trainLen {
		t.Fatalf("seed scores len = %d, want %d", len(state.TrainScores[0]), trainLen)
	}
	if state.BestCandidate != 0 {
		t.Fatalf("BestCandidate = %d, want 0", state.BestCandidate)
	}
}

func TestSetSeedTrainScores_RejectsWrongLength(t *testing.T) {
	state := NewState(testMutationProgram())
	err := SetSeedTrainScores(&state, 3, []float64{1})
	if err == nil {
		t.Fatal("SetSeedTrainScores() error = nil, want error")
	}
}

func TestAddMetricCalls_Accumulates(t *testing.T) {
	state := NewState(testMutationProgram())
	if err := AddMetricCalls(&state, 4); err != nil {
		t.Fatalf("AddMetricCalls() unexpected error: %v", err)
	}
	if err := AddMetricCalls(&state, 2); err != nil {
		t.Fatalf("AddMetricCalls() unexpected error: %v", err)
	}
	if state.MetricCalls != 6 {
		t.Fatalf("MetricCalls = %d, want 6", state.MetricCalls)
	}
}

func TestAddMetricCalls_RejectsNegative(t *testing.T) {
	state := NewState(testMutationProgram())
	if err := AddMetricCalls(&state, -1); err == nil {
		t.Fatal("AddMetricCalls(-1) error = nil, want error")
	}
}

func TestAcceptCandidate_Reflection_AppendsAligned(t *testing.T) {
	prog := testMutationProgram()
	state := NewState(prog)
	trainLen := 2
	if err := SetSeedTrainScores(&state, trainLen, []float64{0, 1}); err != nil {
		t.Fatalf("SetSeedTrainScores() unexpected error: %v", err)
	}
	id, err := AcceptCandidate(&state, trainLen, AcceptCandidateParams{
		ParentIDs:     []int{0},
		ProposalKind:  ProposalReflection,
		MutatedModule: "answer",
		CreatedAtIter: 1,
		Prompts:       Candidate{"answer": "improved"},
		TrainScores:   []float64{1, 1},
	})
	if err != nil {
		t.Fatalf("AcceptCandidate() unexpected error: %v", err)
	}
	if id != 1 {
		t.Fatalf("new ID = %d, want 1", id)
	}
	if len(state.Candidates) != 2 || state.Candidates[1].MutatedModule != "answer" {
		t.Fatalf("candidates = %#v", state.Candidates)
	}
}

func TestAcceptCandidate_PromptsImmutable(t *testing.T) {
	state := NewState(testMutationProgram())
	trainLen := 1
	if err := SetSeedTrainScores(&state, trainLen, []float64{0}); err != nil {
		t.Fatalf("SetSeedTrainScores() unexpected error: %v", err)
	}
	proposal := Candidate{"answer": "v2"}
	id, err := AcceptCandidate(&state, trainLen, AcceptCandidateParams{
		ParentIDs:     []int{0},
		ProposalKind:  ProposalReflection,
		MutatedModule: "answer",
		CreatedAtIter: 1,
		Prompts:       proposal,
		TrainScores:   []float64{1},
	})
	if err != nil {
		t.Fatalf("AcceptCandidate() unexpected error: %v", err)
	}
	proposal["answer"] = "mutated"
	if state.Candidates[id].Prompts["answer"] != "v2" {
		t.Fatalf("stored prompt = %q, want %q (immutable)", state.Candidates[id].Prompts["answer"], "v2")
	}
}

func TestAcceptCandidate_RejectsInvalidParent(t *testing.T) {
	state := NewState(testMutationProgram())
	trainLen := 1
	if err := SetSeedTrainScores(&state, trainLen, []float64{0}); err != nil {
		t.Fatalf("SetSeedTrainScores() unexpected error: %v", err)
	}
	_, err := AcceptCandidate(&state, trainLen, AcceptCandidateParams{
		ParentIDs:     []int{1},
		ProposalKind:  ProposalReflection,
		MutatedModule: "answer",
		CreatedAtIter: 1,
		Prompts:       Candidate{"answer": "v2"},
		TrainScores:   []float64{1},
	})
	if err == nil {
		t.Fatal("AcceptCandidate() error = nil, want error")
	}
}

func TestAcceptCandidate_MergeShape(t *testing.T) {
	state := NewState(testMutationProgram())
	trainLen := 2
	if err := SetSeedTrainScores(&state, trainLen, []float64{0, 0}); err != nil {
		t.Fatalf("SetSeedTrainScores() unexpected error: %v", err)
	}
	_, err := AcceptCandidate(&state, trainLen, AcceptCandidateParams{
		ParentIDs:     []int{0},
		ProposalKind:  ProposalReflection,
		MutatedModule: "answer",
		CreatedAtIter: 1,
		Prompts:       Candidate{"answer": "mid"},
		TrainScores:   []float64{0.5, 0.5},
	})
	if err != nil {
		t.Fatalf("first AcceptCandidate() unexpected error: %v", err)
	}
	id, err := AcceptCandidate(&state, trainLen, AcceptCandidateParams{
		ParentIDs:     []int{0, 1},
		ProposalKind:  ProposalMerge,
		CreatedAtIter: 2,
		Prompts:       Candidate{"answer": "merged"},
		TrainScores:   []float64{1, 1},
	})
	if err != nil {
		t.Fatalf("merge AcceptCandidate() unexpected error: %v", err)
	}
	if id != 2 || state.Candidates[2].ProposalKind != ProposalMerge {
		t.Fatalf("merge candidate = %#v", state.Candidates[2])
	}
}

func TestRecomputeBestCandidate_PicksHighestMean(t *testing.T) {
	state := State{
		Candidates: []CandidateRecord{{ID: 0}, {ID: 1}},
		TrainScores: [][]float64{
			{0, 0},
			{1, 1},
		},
	}
	if err := RecomputeBestCandidate(&state); err != nil {
		t.Fatalf("RecomputeBestCandidate() unexpected error: %v", err)
	}
	if state.BestCandidate != 1 {
		t.Fatalf("BestCandidate = %d, want 1", state.BestCandidate)
	}
}

func TestRecomputeBestCandidate_TieKeepsLowerID(t *testing.T) {
	state := State{
		Candidates: []CandidateRecord{{ID: 0}, {ID: 1}},
		TrainScores: [][]float64{
			{1, 1},
			{1, 1},
		},
	}
	if err := RecomputeBestCandidate(&state); err != nil {
		t.Fatalf("RecomputeBestCandidate() unexpected error: %v", err)
	}
	if state.BestCandidate != 0 {
		t.Fatalf("BestCandidate = %d, want 0 on tie", state.BestCandidate)
	}
}

func TestRecomputeBestCandidate_EmptyPool(t *testing.T) {
	state := State{}
	err := RecomputeBestCandidate(&state)
	if err == nil {
		t.Fatal("RecomputeBestCandidate() error = nil, want error")
	}
	if !errors.Is(err, ErrEmptyCandidatePool) {
		t.Fatalf("error = %v, want ErrEmptyCandidatePool", err)
	}
}

func TestStateScoreRowsAlignWithCandidateIDs(t *testing.T) {
	state := State{
		Candidates: []CandidateRecord{
			{ID: 0, ParentIDs: []int{}, ProposalKind: ProposalSeed},
			{ID: 1, ParentIDs: []int{0}, ProposalKind: ProposalReflection},
		},
		TrainScores: [][]float64{
			{0, 1, 0},
			{1, 1, 0},
		},
		BestCandidate: 1,
	}

	for i, candidate := range state.Candidates {
		if candidate.ID != i {
			t.Fatalf("candidate at index %d has ID %d", i, candidate.ID)
		}
		if !reflect.DeepEqual(state.TrainScores[candidate.ID], state.TrainScores[i]) {
			t.Fatalf("score row for candidate %d does not match row at index %d", candidate.ID, i)
		}
	}
}
