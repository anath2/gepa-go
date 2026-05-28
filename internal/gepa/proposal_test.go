package gepa

import "testing"

func TestMutatedCandidate_ClonesParentAndSetsModule(t *testing.T) {
	parent := Candidate{"retriever": "r", "answer": "seed"}
	got := mutatedCandidate(parent, "answer", "answer v2")
	if got["retriever"] != "r" || got["answer"] != "answer v2" {
		t.Fatalf("mutatedCandidate() = %#v", got)
	}
	parent["answer"] = "mutated"
	if got["answer"] != "answer v2" {
		t.Fatal("mutatedCandidate should not alias parent prompts")
	}
}
