package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeInspectFixtureRun(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "run")
	if err := os.MkdirAll(filepath.Join(dir, "candidates"), 0o755); err != nil {
		t.Fatalf("mkdir candidates: %v", err)
	}

	state := map[string]any{
		"iteration":      2,
		"metric_calls":   12,
		"best_candidate": 1,
		"candidates": []map[string]any{
			{
				"id": 0, "parent_ids": []int{}, "proposal_kind": "seed",
				"created_at_iter": 0, "prompts": map[string]string{"answer": "seed prompt"},
			},
			{
				"id": 1, "parent_ids": []int{0}, "proposal_kind": "reflection",
				"mutated_module": "answer", "created_at_iter": 1,
				"prompts": map[string]string{"answer": "improved prompt"},
			},
		},
		"train_scores": [][]float64{{0.5, 1.0}, {0.75, 1.0}},
	}
	writeJSONFile(t, filepath.Join(dir, "state.json"), state)

	events := []string{
		`{"type":"seed_evaluated","iteration":0,"metric_calls":6,"candidate_id":0,"proposal_kind":"seed"}`,
		`{"type":"proposal_requested","iteration":1,"metric_calls":9,"candidate_id":0,"parent_ids":[0],"proposal_kind":"reflection","mutated_module":"answer","batch_indices":[0,1,2]}`,
		`{"type":"candidate_accepted","iteration":1,"metric_calls":12,"candidate_id":1,"parent_ids":[0],"proposal_kind":"reflection","mutated_module":"answer","parent_mean":0.25,"proposal_mean":0.75,"accepted":true}`,
	}
	if err := os.WriteFile(filepath.Join(dir, "events.jsonl"), []byte(strings.Join(events, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write events.jsonl: %v", err)
	}

	result := map[string]any{
		"best_candidate": 1,
		"metric_calls":   12,
		"train_mean":     0.875,
	}
	writeJSONFile(t, filepath.Join(dir, "result.json"), result)

	return dir
}

func writeJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal %s: %v", path, err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestInspectTextOutput(t *testing.T) {
	dir := writeInspectFixtureRun(t)
	out, _, err := runCmd(t, "inspect", dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{
		"run:",
		"iteration=2",
		"metric_calls=12",
		"best_candidate=1",
		"train_mean=0.875",
		"candidates:",
		"0000 seed",
		"0001 reflection",
		"parent 0000",
		"events:",
		"seed_evaluated",
		"proposal_requested",
		"candidate_accepted",
		"improved prompt",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in output:\n%s", want, out)
		}
	}
}

func TestInspectJSONOutput(t *testing.T) {
	dir := writeInspectFixtureRun(t)
	out, _, err := runCmd(t, "inspect", dir, "--format", "json")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var got inspectReport
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("decode inspect json: %v\noutput:\n%s", err, out)
	}
	if got.RunDir != dir {
		t.Errorf("run_dir = %q, want %q", got.RunDir, dir)
	}
	if got.Summary.BestCandidate != 1 || got.Summary.MetricCalls != 12 {
		t.Fatalf("summary = %#v", got.Summary)
	}
	if len(got.Candidates) != 2 || len(got.Events) != 3 {
		t.Fatalf("candidates=%d events=%d", len(got.Candidates), len(got.Events))
	}
}

func TestInspectShowTreeFalse(t *testing.T) {
	dir := writeInspectFixtureRun(t)
	out, _, err := runCmd(t, "inspect", dir, "--show-tree=false")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(out, "candidates:") {
		t.Errorf("expected no candidate tree, got:\n%s", out)
	}
	if !strings.Contains(out, "events:") {
		t.Errorf("expected events section, got:\n%s", out)
	}
}

func TestInspectShowEventsFalse(t *testing.T) {
	dir := writeInspectFixtureRun(t)
	out, _, err := runCmd(t, "inspect", dir, "--show-events=false")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(out, "events:") {
		t.Errorf("expected no events section, got:\n%s", out)
	}
	if !strings.Contains(out, "candidates:") {
		t.Errorf("expected candidate tree, got:\n%s", out)
	}
}

func TestInspectMissingState(t *testing.T) {
	dir := t.TempDir()
	_, _, err := runCmd(t, "inspect", dir)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "state.json") {
		t.Errorf("got %q, want state.json mention", err.Error())
	}
}

func TestInspectMalformedEvents(t *testing.T) {
	dir := writeInspectFixtureRun(t)
	if err := os.WriteFile(filepath.Join(dir, "events.jsonl"), []byte("{not json}\n"), 0o644); err != nil {
		t.Fatalf("write bad events: %v", err)
	}
	_, _, err := runCmd(t, "inspect", dir)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "events.jsonl") {
		t.Errorf("got %q, want events.jsonl mention", err.Error())
	}
}

func TestInspectEmptyDir(t *testing.T) {
	dir := t.TempDir()
	_, _, err := runCmd(t, "inspect", dir)
	if err == nil {
		t.Fatal("expected error for empty run dir")
	}
}
