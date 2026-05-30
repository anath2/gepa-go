package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/gepa"
)

var runE2E = flag.Bool("run-e2e", false, "run high-latency end-to-end tests")

// TestOptimizeChineseNERLoopProgressE2E is a red-green milestone test for the
// live optimization loop. It documents how far a real run progresses today:
// seed evaluation, at least one proposal attempt, reflection soft-failure
// (reflector stub), and final result persistence — but no accepted candidates
// until ReflectionProposer lands.
func TestOptimizeChineseNERLoopProgressE2E(t *testing.T) {
	if !*runE2E {
		t.Skip("set -run-e2e to run high-latency end-to-end tests")
	}
	if strings.TrimSpace(os.Getenv("API_KEY")) == "" {
		t.Skip("API_KEY required for live e2e")
	}
	if strings.TrimSpace(os.Getenv("BASE_URL")) == "" {
		t.Skip("BASE_URL required for live e2e")
	}

	fixtureDir := filepath.Join("..", "..", "internal", "testdata", "chinese_ner")
	programPath := filepath.Join(fixtureDir, "program.json")
	trainPath := filepath.Join(fixtureDir, "train.jsonl")
	valPath := filepath.Join(fixtureDir, "val.jsonl")

	runParent := t.TempDir()
	configPath := writeChineseNERE2EConfig(t, runParent)

	validateChineseNERE2EConfig(t, programPath, configPath, trainPath, valPath)
	trainLen := 6

	const runID = "chinese-ner-e2e"
	out, _, err := runCmd(t, "optimize",
		"--program", programPath,
		"--config", configPath,
		"--train", trainPath,
		"--val", valPath,
		"--run-id", runID,
	)
	if err != nil {
		t.Fatalf("optimize failed: %v\nstdout:\n%s", err, out)
	}

	for _, want := range []string{
		"program:  3 modules, 0 tools",
		`config:   budget=12 minibatch=3 seed=20260522`,
		"run:",
		"best:     candidate",
		"metric_calls=",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing stdout milestone %q\nfull output:\n%s", want, out)
		}
	}

	runDir := filepath.Join(runParent, runID)
	for _, name := range []string{"program.json", "config.json", "state.json", "result.json", "events.jsonl", "candidates/0000.json"} {
		if _, err := os.Stat(filepath.Join(runDir, name)); err != nil {
			t.Fatalf("missing run artifact %s: %v", name, err)
		}
	}

	events := readRunEvents(t, filepath.Join(runDir, "events.jsonl"))
	eventTypes := eventTypeSet(events)
	for _, want := range []string{"seed_evaluated", "proposal_requested", "proposal_failed"} {
		if !eventTypes[want] {
			t.Fatalf("events missing %q; got types %v", want, eventTypes)
		}
	}
	if eventTypes["candidate_accepted"] {
		t.Fatal("candidate_accepted present; update this test when ReflectionProposer is wired")
	}

	var result gepa.Result
	resultData, err := os.ReadFile(filepath.Join(runDir, "result.json"))
	if err != nil {
		t.Fatalf("read result.json: %v", err)
	}
	if err := json.Unmarshal(resultData, &result); err != nil {
		t.Fatalf("decode result.json: %v", err)
	}
	if result.MetricCalls < trainLen {
		t.Fatalf("metric_calls = %d, want at least seed eval (%d train examples)", result.MetricCalls, trainLen)
	}

	fmt.Fprintf(os.Stdout, "e2e progression: run=%s metric_calls=%d best=%d train_mean=%.4g events=%v\n",
		runDir, result.MetricCalls, result.BestCandidate, result.TrainMean, eventTypes)
}

func writeChineseNERE2EConfig(t *testing.T, logDir string) string {
	t.Helper()
	fixtureConfig := filepath.Join("..", "..", "internal", "testdata", "chinese_ner", "config.json")
	data, err := os.ReadFile(fixtureConfig)
	if err != nil {
		t.Fatalf("read fixture config: %v", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("decode fixture config: %v", err)
	}
	cfg["log_dir"] = logDir
	cfg["log_traces"] = false

	out, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatalf("encode e2e config: %v", err)
	}
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write e2e config: %v", err)
	}
	return path
}

func validateChineseNERE2EConfig(t *testing.T, programPath, configPath, trainPath, valPath string) {
	t.Helper()

	fmt.Fprintf(os.Stdout, "e2e validation: loading problem from %s\n", programPath)
	problem, err := gepa.LoadProblem(gepa.ProblemPaths{
		Program: programPath,
		Config:  configPath,
		Train:   trainPath,
		Val:     valPath,
	})
	if err != nil {
		t.Fatalf("problem validation failed: %v", err)
	}
	fmt.Fprintf(os.Stdout, "e2e validation: problem ok (%d modules, %d tools, train=%d val=%d)\n",
		len(problem.Program.Modules), len(problem.Program.Tools), len(problem.Train), len(problem.Val))
}

type runEvent struct {
	Type string `json:"type"`
}

func readRunEvents(t *testing.T, path string) []runEvent {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open events: %v", err)
	}
	defer f.Close()

	var events []runEvent
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var ev runEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("decode event line: %v", err)
		}
		events = append(events, ev)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan events: %v", err)
	}
	return events
}

func eventTypeSet(events []runEvent) map[string]bool {
	out := make(map[string]bool, len(events))
	for _, ev := range events {
		out[ev.Type] = true
	}
	return out
}
