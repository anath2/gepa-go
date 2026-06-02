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
	paths := newRunArtifacts(runDir)

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
	if paths.TrajectoriesDir != filepath.Join(runDir, "trajectories") {
		t.Fatalf("TrajectoriesDir = %q", paths.TrajectoriesDir)
	}
	if paths.ResultPath != filepath.Join(runDir, "result.json") {
		t.Fatalf("ResultPath = %q", paths.ResultPath)
	}
}

func TestEnsureRunDir_CreatesRunAndCandidatesDirs(t *testing.T) {
	paths := newRunArtifacts(filepath.Join(t.TempDir(), "run"))
	if err := ensureRunDir(paths, false); err != nil {
		t.Fatalf("ensureRunDir() unexpected error: %v", err)
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

func TestEnsureRunDir_CreatesTrajectoriesDirWhenEnabled(t *testing.T) {
	paths := newRunArtifacts(filepath.Join(t.TempDir(), "run"))
	if err := ensureRunDir(paths, true); err != nil {
		t.Fatalf("ensureRunDir() unexpected error: %v", err)
	}
	info, err := os.Stat(paths.TrajectoriesDir)
	if err != nil {
		t.Fatalf("Stat(%q) error: %v", paths.TrajectoriesDir, err)
	}
	if !info.IsDir() {
		t.Fatalf("%q is not a directory", paths.TrajectoriesDir)
	}
}

func TestWriteState_WritesValidJSON(t *testing.T) {
	paths := newRunArtifacts(t.TempDir())
	if err := ensureRunDir(paths, false); err != nil {
		t.Fatalf("ensureRunDir() unexpected error: %v", err)
	}

	want := poolState{
		Iteration:     2,
		MetricCalls:   10,
		BestCandidate: 0,
		Candidates: []candidateRecord{
			{ID: 0, ParentIDs: []int{}, ProposalKind: proposalSeed, Prompts: Candidate{"answer": "seed"}},
		},
		TrainScores: [][]float64{{0.5, 1}},
	}
	if err := writeState(paths, want); err != nil {
		t.Fatalf("writeState() unexpected error: %v", err)
	}

	var got poolState
	if err := readJSONFile(paths.StatePath, &got); err != nil {
		t.Fatalf("read state.json: %v", err)
	}
	if got.Iteration != want.Iteration || got.MetricCalls != want.MetricCalls {
		t.Fatalf("state = %#v, want iteration/metric_calls from %#v", got, want)
	}
	if len(got.Candidates) != 1 || got.Candidates[0].ProposalKind != proposalSeed {
		t.Fatalf("candidates = %#v", got.Candidates)
	}
}

func TestWriteState_OverwritesExisting(t *testing.T) {
	paths := newRunArtifacts(t.TempDir())
	if err := ensureRunDir(paths, false); err != nil {
		t.Fatalf("ensureRunDir() unexpected error: %v", err)
	}
	if err := writeState(paths, poolState{Iteration: 1, MetricCalls: 1, BestCandidate: 0, Candidates: []candidateRecord{{ID: 0}}, TrainScores: [][]float64{{0}}}); err != nil {
		t.Fatalf("first writeState() unexpected error: %v", err)
	}
	if err := writeState(paths, poolState{Iteration: 9, MetricCalls: 99, BestCandidate: 0, Candidates: []candidateRecord{{ID: 0}}, TrainScores: [][]float64{{1}}}); err != nil {
		t.Fatalf("second writeState() unexpected error: %v", err)
	}

	var got poolState
	if err := readJSONFile(paths.StatePath, &got); err != nil {
		t.Fatalf("read state.json: %v", err)
	}
	if got.Iteration != 9 || got.MetricCalls != 99 {
		t.Fatalf("state = %#v, want updated iteration and metric_calls", got)
	}
}

func TestAppendEvent_WritesOneJSONPerLine(t *testing.T) {
	paths := newRunArtifacts(t.TempDir())
	accepted := true
	parentMean := 0.25
	proposalMean := 0.75
	event := eventRecord{
		Type:         "candidate_accepted",
		Iteration:    1,
		MetricCalls:  5,
		CandidateID:  1,
		ParentIDs:    []int{0},
		ProposalKind: proposalReflection,
		MutatedModule: "answer",
		ParentMean:   &parentMean,
		ProposalMean: &proposalMean,
		Accepted:     &accepted,
	}
	if err := appendEvent(paths, event); err != nil {
		t.Fatalf("appendEvent() unexpected error: %v", err)
	}

	lines, err := readJSONLLines(paths.EventsPath)
	if err != nil {
		t.Fatalf("read events.jsonl: %v", err)
	}
	if len(lines) != 1 {
		t.Fatalf("got %d lines, want 1", len(lines))
	}
	var got eventRecord
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		t.Fatalf("unmarshal event: %v", err)
	}
	if got.Type != event.Type || got.CandidateID != 1 {
		t.Fatalf("event = %#v, want type and candidate_id from %#v", got, event)
	}
}

func TestAppendEvent_AppendsMultipleLines(t *testing.T) {
	paths := newRunArtifacts(t.TempDir())
	for i, typ := range []string{"seed_evaluated", "proposal_evaluated", "candidate_rejected"} {
		if err := appendEvent(paths, eventRecord{Type: typ, Iteration: i, MetricCalls: i + 1}); err != nil {
			t.Fatalf("appendEvent(%q) unexpected error: %v", typ, err)
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
	paths := newRunArtifacts(t.TempDir())
	if err := ensureRunDir(paths, false); err != nil {
		t.Fatalf("ensureRunDir() unexpected error: %v", err)
	}

	record := candidateRecord{
		ID:            7,
		ParentIDs:     []int{0},
		ProposalKind:  proposalReflection,
		MutatedModule: "answer",
		CreatedAtIter: 3,
		Prompts:       Candidate{"answer": "v7"},
	}
	if err := writeCandidate(paths, 7, record); err != nil {
		t.Fatalf("writeCandidate() unexpected error: %v", err)
	}

	wantPath := filepath.Join(paths.CandidatesDir, "0007.json")
	if _, err := os.Stat(wantPath); err != nil {
		t.Fatalf("Stat(%q) error: %v", wantPath, err)
	}

	var got candidateRecord
	if err := readJSONFile(wantPath, &got); err != nil {
		t.Fatalf("read candidate: %v", err)
	}
	if got.ID != 7 || got.Prompts["answer"] != "v7" {
		t.Fatalf("record = %#v, want id 7 and prompt v7", got)
	}
}

func TestWriteResult_WritesValidJSON(t *testing.T) {
	paths := newRunArtifacts(t.TempDir())
	valMean := 0.9
	result := Result{
		BestCandidate: 1,
		MetricCalls:   42,
		TrainMean:     0.8,
		ValidationMean: &valMean,
	}
	if err := writeResult(paths, result); err != nil {
		t.Fatalf("writeResult() unexpected error: %v", err)
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

func TestRunWriter_ProposalEvents(t *testing.T) {
	runDir := t.TempDir()
	writer := newRunWriter(runDir, false)
	if err := writer.init(); err != nil {
		t.Fatalf("init() unexpected error: %v", err)
	}

	state := poolState{
		Iteration:   1,
		MetricCalls: 5,
		Candidates:  []candidateRecord{{ID: 0}},
		TrainScores: [][]float64{{1}},
	}
	parentID := 0
	moduleName := "answer"
	batchIndices := []int{0, 2}
	parentMean := 0.25
	proposalMean := 0.75

	if err := writer.proposalRequested(state, parentID, moduleName, batchIndices); err != nil {
		t.Fatalf("proposalRequested() unexpected error: %v", err)
	}
	if err := writer.proposalFailed(state, parentID, moduleName, batchIndices, "reflection failed"); err != nil {
		t.Fatalf("proposalFailed() unexpected error: %v", err)
	}
	if err := writer.proposalEvaluated(state, parentID, moduleName, batchIndices, parentMean, proposalMean); err != nil {
		t.Fatalf("proposalEvaluated() unexpected error: %v", err)
	}
	if err := writer.proposalRejected(state, parentID, moduleName, batchIndices, parentMean, proposalMean); err != nil {
		t.Fatalf("proposalRejected() unexpected error: %v", err)
	}

	events, err := readEventsFromPath(t, newRunArtifacts(runDir).EventsPath)
	if err != nil {
		t.Fatalf("read events: %v", err)
	}
	wantTypes := []string{
		eventProposalRequested,
		eventProposalFailed,
		eventProposalEvaluated,
		eventCandidateRejected,
	}
	if len(events) != len(wantTypes) {
		t.Fatalf("got %d events, want %d", len(events), len(wantTypes))
	}
	for i, typ := range wantTypes {
		if events[i].Type != typ {
			t.Fatalf("events[%d].Type = %q, want %q", i, events[i].Type, typ)
		}
	}
	if events[1].Reason != "reflection failed" {
		t.Fatalf("proposal_failed reason = %q, want reflection failed", events[1].Reason)
	}
	if events[3].Reason != rejectReasonNoImprovement {
		t.Fatalf("candidate_rejected reason = %q, want %q", events[3].Reason, rejectReasonNoImprovement)
	}
	rejected := false
	if events[3].Accepted == nil || *events[3].Accepted != rejected {
		t.Fatalf("candidate_rejected accepted = %v, want false", events[3].Accepted)
	}
	if events[2].ParentMean == nil || *events[2].ParentMean != parentMean {
		t.Fatalf("proposal_evaluated parent_mean = %v, want %v", events[2].ParentMean, parentMean)
	}
}

func readEventsFromPath(t *testing.T, path string) ([]eventRecord, error) {
	t.Helper()
	lines, err := readJSONLLines(path)
	if err != nil {
		return nil, err
	}
	events := make([]eventRecord, 0, len(lines))
	for _, line := range lines {
		var ev eventRecord
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return nil, err
		}
		events = append(events, ev)
	}
	return events, nil
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
