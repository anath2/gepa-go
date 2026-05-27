package gepa

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewRunArtifacts_StandardPaths(t *testing.T) {
	runDir := filepath.Join("runs", "test-run")
	paths := NewRunArtifacts(runDir)

	if paths.RunDir != runDir {
		t.Fatalf("RunDir = %q, want %q", paths.RunDir, runDir)
	}
	if paths.StatePath != filepath.Join(runDir, "state.json") {
		t.Fatalf("StatePath = %q", paths.StatePath)
	}
	if paths.EventsPath != filepath.Join(runDir, "events.jsonl") {
		t.Fatalf("EventsPath = %q", paths.EventsPath)
	}
	if paths.CandidatesDir != filepath.Join(runDir, "candidates") {
		t.Fatalf("CandidatesDir = %q", paths.CandidatesDir)
	}
	if paths.ResultPath != filepath.Join(runDir, "result.json") {
		t.Fatalf("ResultPath = %q", paths.ResultPath)
	}
}

func TestEnsureRunDir_CreatesRunAndCandidatesDirs(t *testing.T) {
	paths := NewRunArtifacts(filepath.Join(t.TempDir(), "run"))
	if err := EnsureRunDir(paths); err != nil {
		t.Fatalf("EnsureRunDir() unexpected error: %v", err)
	}
	for _, dir := range []string{paths.RunDir, paths.CandidatesDir} {
		info, err := os.Stat(dir)
		if err != nil {
			t.Fatalf("Stat(%q) error: %v", dir, err)
		}
		if !info.IsDir() {
			t.Fatalf("%q is not a directory", dir)
		}
	}
}

func TestWriteState_WritesValidJSON(t *testing.T) {
	paths := NewRunArtifacts(t.TempDir())
	if err := EnsureRunDir(paths); err != nil {
		t.Fatalf("EnsureRunDir() unexpected error: %v", err)
	}

	want := State{
		Iteration:     2,
		MetricCalls:   10,
		BestCandidate: 0,
		Candidates: []CandidateRecord{
			{ID: 0, ParentIDs: []int{}, ProposalKind: ProposalSeed, Prompts: Candidate{"answer": "seed"}},
		},
		TrainScores: [][]float64{{0.5, 1}},
	}
	if err := WriteState(paths, want); err != nil {
		t.Fatalf("WriteState() unexpected error: %v", err)
	}

	var got State
	if err := readJSONFile(paths.StatePath, &got); err != nil {
		t.Fatalf("read state.json: %v", err)
	}
	if got.Iteration != want.Iteration || got.MetricCalls != want.MetricCalls {
		t.Fatalf("state = %#v, want iteration/metric_calls from %#v", got, want)
	}
	if len(got.Candidates) != 1 || got.Candidates[0].ProposalKind != ProposalSeed {
		t.Fatalf("candidates = %#v", got.Candidates)
	}
}

func TestWriteState_OverwritesExisting(t *testing.T) {
	paths := NewRunArtifacts(t.TempDir())
	if err := EnsureRunDir(paths); err != nil {
		t.Fatalf("EnsureRunDir() unexpected error: %v", err)
	}
	if err := WriteState(paths, State{Iteration: 1, MetricCalls: 1, BestCandidate: 0, Candidates: []CandidateRecord{{ID: 0}}, TrainScores: [][]float64{{0}}}); err != nil {
		t.Fatalf("first WriteState() unexpected error: %v", err)
	}
	if err := WriteState(paths, State{Iteration: 9, MetricCalls: 99, BestCandidate: 0, Candidates: []CandidateRecord{{ID: 0}}, TrainScores: [][]float64{{1}}}); err != nil {
		t.Fatalf("second WriteState() unexpected error: %v", err)
	}

	var got State
	if err := readJSONFile(paths.StatePath, &got); err != nil {
		t.Fatalf("read state.json: %v", err)
	}
	if got.Iteration != 9 || got.MetricCalls != 99 {
		t.Fatalf("state = %#v, want updated iteration and metric_calls", got)
	}
}

func TestAppendEvent_WritesOneJSONPerLine(t *testing.T) {
	paths := NewRunArtifacts(t.TempDir())
	accepted := true
	parentMean := 0.25
	proposalMean := 0.75
	event := Event{
		Type:         "candidate_accepted",
		Iteration:    1,
		MetricCalls:  5,
		CandidateID:  1,
		ParentIDs:    []int{0},
		ProposalKind: ProposalReflection,
		MutatedModule: "answer",
		ParentMean:   &parentMean,
		ProposalMean: &proposalMean,
		Accepted:     &accepted,
	}
	if err := AppendEvent(paths, event); err != nil {
		t.Fatalf("AppendEvent() unexpected error: %v", err)
	}

	lines, err := readJSONLLines(paths.EventsPath)
	if err != nil {
		t.Fatalf("read events.jsonl: %v", err)
	}
	if len(lines) != 1 {
		t.Fatalf("got %d lines, want 1", len(lines))
	}
	var got Event
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		t.Fatalf("unmarshal event: %v", err)
	}
	if got.Type != event.Type || got.CandidateID != 1 {
		t.Fatalf("event = %#v, want type and candidate_id from %#v", got, event)
	}
}

func TestAppendEvent_AppendsMultipleLines(t *testing.T) {
	paths := NewRunArtifacts(t.TempDir())
	for i, typ := range []string{"seed_evaluated", "proposal_evaluated", "candidate_rejected"} {
		if err := AppendEvent(paths, Event{Type: typ, Iteration: i, MetricCalls: i + 1}); err != nil {
			t.Fatalf("AppendEvent(%q) unexpected error: %v", typ, err)
		}
	}

	lines, err := readJSONLLines(paths.EventsPath)
	if err != nil {
		t.Fatalf("read events.jsonl: %v", err)
	}
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3", len(lines))
	}
	for i, typ := range []string{"seed_evaluated", "proposal_evaluated", "candidate_rejected"} {
		if !strings.Contains(lines[i], typ) {
			t.Fatalf("line %d = %q, want type %q", i, lines[i], typ)
		}
	}
}

func TestWriteCandidate_ZeroPaddedFilename(t *testing.T) {
	paths := NewRunArtifacts(t.TempDir())
	if err := EnsureRunDir(paths); err != nil {
		t.Fatalf("EnsureRunDir() unexpected error: %v", err)
	}

	record := CandidateRecord{
		ID:            7,
		ParentIDs:     []int{0},
		ProposalKind:  ProposalReflection,
		MutatedModule: "answer",
		CreatedAtIter: 3,
		Prompts:       Candidate{"answer": "v7"},
	}
	if err := WriteCandidate(paths, 7, record); err != nil {
		t.Fatalf("WriteCandidate() unexpected error: %v", err)
	}

	wantPath := filepath.Join(paths.CandidatesDir, "0007.json")
	if _, err := os.Stat(wantPath); err != nil {
		t.Fatalf("Stat(%q) error: %v", wantPath, err)
	}

	var got CandidateRecord
	if err := readJSONFile(wantPath, &got); err != nil {
		t.Fatalf("read candidate: %v", err)
	}
	if got.ID != 7 || got.Prompts["answer"] != "v7" {
		t.Fatalf("record = %#v, want id 7 and prompt v7", got)
	}
}

func TestWriteResult_WritesValidJSON(t *testing.T) {
	paths := NewRunArtifacts(t.TempDir())
	valMean := 0.9
	result := Result{
		BestCandidate: 1,
		MetricCalls:   42,
		TrainMean:     0.8,
		ValidationMean: &valMean,
	}
	if err := WriteResult(paths, result); err != nil {
		t.Fatalf("WriteResult() unexpected error: %v", err)
	}

	var got Result
	if err := readJSONFile(paths.ResultPath, &got); err != nil {
		t.Fatalf("read result.json: %v", err)
	}
	if got.BestCandidate != 1 || got.MetricCalls != 42 || got.TrainMean != 0.8 {
		t.Fatalf("result = %#v, want best/metric/train mean from %#v", got, result)
	}
	if got.ValidationMean == nil || *got.ValidationMean != 0.9 {
		t.Fatalf("ValidationMean = %v, want 0.9", got.ValidationMean)
	}
}

func readJSONFile(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

func readJSONLLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	return lines, sc.Err()
}
