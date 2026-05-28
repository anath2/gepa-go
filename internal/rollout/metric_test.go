package rollout

import (
	"testing"
)

func TestExactMatchHit(t *testing.T) {
	result := scoreExactMatch(
		map[string]any{"answer": "Paris"},
		map[string]any{"answer": "Paris"},
		"answer",
	)
	if result.Score != 1 {
		t.Fatalf("Score = %v, want 1", result.Score)
	}
	if result.Feedback == "" {
		t.Fatal("Feedback = empty, want non-empty confirmation")
	}
}

func TestExactMatchMiss(t *testing.T) {
	result := scoreExactMatch(
		map[string]any{"answer": "London"},
		map[string]any{"answer": "Paris"},
		"answer",
	)
	if result.Score != 0 {
		t.Fatalf("Score = %v, want 0", result.Score)
	}
	want := `expected "Paris", got "London"`
	if result.Feedback != want {
		t.Fatalf("Feedback = %q, want %q", result.Feedback, want)
	}
}

func TestExactMatchMissingField(t *testing.T) {
	result := scoreExactMatch(
		map[string]any{"other": "value"},
		map[string]any{"answer": "Paris"},
		"answer",
	)
	if result.Score != 0 {
		t.Fatalf("Score = %v, want 0", result.Score)
	}
	want := `expected "Paris", got <missing>`
	if result.Feedback != want {
		t.Fatalf("Feedback = %q, want %q", result.Feedback, want)
	}
	if result.Error == "" {
		t.Fatal("Error = empty, want missing-field reason")
	}
}
