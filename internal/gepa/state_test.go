package gepa

import (
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
