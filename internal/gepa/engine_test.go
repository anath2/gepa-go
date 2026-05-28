package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/program"
)

func TestOptionsHoldsInputsAndDependencies(t *testing.T) {
	prog := program.Program{Modules: []program.Module{{Name: "answer", Prompt: "answer"}}}
	cfg := engineConfig(10, 3, 7)
	train := []program.Example{{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "a"}}}
	val := []program.Example{{Input: map[string]any{"question": "vq"}, Expected: map[string]any{"answer": "va"}}}

	opts := Options{
		Program:   prog,
		Config:    cfg,
		Train:     train,
		Val:       val,
		RunDir:    "runs/test",
		LogTraces: true,
		Evaluator: fakeEvaluator{},
		Reflector: fakeReflector{},
	}

	if len(opts.Program.Modules) != 1 {
		t.Fatalf("Program modules = %d, want 1", len(opts.Program.Modules))
	}
	if opts.Config.Seed != 7 {
		t.Fatalf("Config seed = %d, want 7", opts.Config.Seed)
	}
	if len(opts.Train) != 1 || len(opts.Val) != 1 {
		t.Fatalf("train/val lengths = %d/%d, want 1/1", len(opts.Train), len(opts.Val))
	}
	if opts.Evaluator == nil || opts.Reflector == nil {
		t.Fatal("expected dependencies to be assignable on Options")
	}
}

func TestOptimizeStubInstallsDefaultsAndReturnsNotImplemented(t *testing.T) {
	opts := Options{
		Program: program.Program{Modules: []program.Module{{Name: "answer", Prompt: "answer"}}},
		Config:  engineConfig(10, 3, 7),
		Train:   []program.Example{{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "a"}}},
	}

	_, err := Optimize(context.Background(), opts)
	if err == nil {
		t.Fatal("Optimize() error = nil, want not implemented")
	}
	if !errors.Is(err, errEvaluatorNotImplemented) {
		t.Fatalf("Optimize() error = %v, want errEvaluatorNotImplemented", err)
	}
}

func TestDefaultEvaluatorReturnsStableNotImplementedError(t *testing.T) {
	results, err := defaultEvaluator{}.Evaluate(context.Background(), Candidate{"answer": "prompt"}, nil)
	if err == nil {
		t.Fatal("Evaluate() error = nil, want not implemented")
	}
	if results != nil {
		t.Fatalf("Evaluate() results = %#v, want nil", results)
	}
	if !errors.Is(err, errEvaluatorNotImplemented) {
		t.Fatalf("Evaluate() error = %v, want errEvaluatorNotImplemented", err)
	}
}

func TestDefaultReflectorReturnsStableNotImplementedError(t *testing.T) {
	proposal, err := defaultReflector{}.Propose(context.Background(), ReflectionRequest{
		Candidate:  Candidate{"answer": "prompt"},
		ParentID:   0,
		ModuleName: "answer",
	})
	if err == nil {
		t.Fatal("Propose() error = nil, want not implemented")
	}
	if proposal != "" {
		t.Fatalf("Propose() proposal = %q, want empty", proposal)
	}
	if !errors.Is(err, errReflectorNotImplemented) {
		t.Fatalf("Propose() error = %v, want errReflectorNotImplemented", err)
	}
}

func TestOptimizeEvaluatesSeedOnFullTrain(t *testing.T) {
	prog := twoModuleProgram()
	train := makeTrainExamples(4)
	opts := baseOpts(prog, train, engineConfig(20, 2, 1))
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor: func(Candidate, []program.Example) float64 { return 0.25 },
	}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	result, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	if result.MetricCalls < len(train) {
		t.Fatalf("MetricCalls = %d, want at least %d", result.MetricCalls, len(train))
	}
	if result.BestCandidate != 0 {
		t.Fatalf("BestCandidate = %d, want 0", result.BestCandidate)
	}
	if result.TrainMean != 0.25 {
		t.Fatalf("TrainMean = %v, want 0.25", result.TrainMean)
	}
}

func TestOptimizeCountsBudgetByMetricCallsNotIterations(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(3)
	opts := baseOpts(prog, train, engineConfig(3, 1, 2))
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	result, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	if result.MetricCalls != 3 {
		t.Fatalf("MetricCalls = %d, want 3 (seed only)", result.MetricCalls)
	}
}

func TestOptimizeStopsBeforeBudgetOverflow(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	opts := baseOpts(prog, train, engineConfig(len(train), 2, 3))
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	result, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	if result.MetricCalls != len(train) {
		t.Fatalf("MetricCalls = %d, want %d (no proposal work)", result.MetricCalls, len(train))
	}
}

func TestOptimizeAcceptsStrictMinibatchImprovement(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	opts := baseOpts(prog, train, engineConfig(20, 2, 4))
	trainLen := len(train)
	opts.Evaluator = &scriptedEvaluator{
		trainSize: trainLen,
		scoreFor: func(c Candidate, examples []program.Example) float64 {
			if len(examples) < trainLen {
				if c["answer"] == "answer v2" {
					return 1
				}
				return 0
			}
			if c["answer"] == "answer v2" {
				return 1
			}
			return 0.25
		},
	}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	result, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	if result.BestCandidate == 0 {
		t.Fatal("BestCandidate still seed, want accepted reflection candidate")
	}
}

func TestOptimizeRejectsEqualOrLowerProposal(t *testing.T) {
	prog := twoModuleProgram()
	train := makeTrainExamples(4)
	opts := baseOpts(prog, train, engineConfig(20, 2, 5))
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{proposal: "answer seed"}

	result, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	if result.BestCandidate != 0 {
		t.Fatalf("BestCandidate = %d, want seed candidate 0", result.BestCandidate)
	}
}

func TestOptimizeHandlesReflectionFailureWithoutNewCandidate(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	opts := baseOpts(prog, train, engineConfig(20, 2, 6))
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{err: errors.New("reflection failed")}

	result, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	if result.BestCandidate != 0 {
		t.Fatalf("BestCandidate = %d, want seed after reflection failure", result.BestCandidate)
	}
	if result.MetricCalls < len(train)+2 {
		t.Fatalf("MetricCalls = %d, want parent minibatch counted", result.MetricCalls)
	}
}

func TestOptimizePersistsSeedAndResultWhenRunDirSet(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	runDir := t.TempDir()
	opts := baseOpts(prog, train, engineConfig(len(train), 2, 8))
	opts.Val = nil
	opts.RunDir = runDir
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 0.5 },
	}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	result, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}

	paths := newRunArtifacts(runDir)
	assertFileExists(t, paths.StatePath)
	assertFileExists(t, filepath.Join(paths.CandidatesDir, "0000.json"))
	assertFileExists(t, paths.ResultPath)

	var state poolState
	if err := readJSONFile(paths.StatePath, &state); err != nil {
		t.Fatalf("read state.json: %v", err)
	}
	if state.MetricCalls != len(train) || state.BestCandidate != 0 {
		t.Fatalf("state = %#v, want seed-only pool", state)
	}

	var seed candidateRecord
	if err := readJSONFile(filepath.Join(paths.CandidatesDir, "0000.json"), &seed); err != nil {
		t.Fatalf("read candidates/0000.json: %v", err)
	}
	if seed.ProposalKind != proposalSeed || seed.Prompts["answer"] != "answer seed" {
		t.Fatalf("seed record = %#v", seed)
	}

	events, err := readEvents(t, paths.EventsPath)
	if err != nil {
		t.Fatalf("read events: %v", err)
	}
	if len(events) != 1 || events[0].Type != eventSeedEvaluated {
		t.Fatalf("events = %#v, want single seed_evaluated", events)
	}

	var gotResult Result
	if err := readJSONFile(paths.ResultPath, &gotResult); err != nil {
		t.Fatalf("read result.json: %v", err)
	}
	if gotResult.BestCandidate != result.BestCandidate || gotResult.TrainMean != result.TrainMean {
		t.Fatalf("result.json = %#v, want %#v", gotResult, result)
	}
}

func TestOptimizePersistsAcceptedCandidate(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	runDir := t.TempDir()
	trainLen := len(train)
	opts := baseOpts(prog, train, engineConfig(20, 2, 9))
	opts.Val = nil
	opts.RunDir = runDir
	opts.Evaluator = &scriptedEvaluator{
		trainSize: trainLen,
		scoreFor: func(c Candidate, examples []program.Example) float64 {
			if len(examples) < trainLen {
				if c["answer"] == "answer v2" {
					return 1
				}
				return 0
			}
			if c["answer"] == "answer v2" {
				return 1
			}
			return 0.25
		},
	}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	_, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}

	paths := newRunArtifacts(runDir)
	assertFileExists(t, filepath.Join(paths.CandidatesDir, "0001.json"))

	var accepted candidateRecord
	if err := readJSONFile(filepath.Join(paths.CandidatesDir, "0001.json"), &accepted); err != nil {
		t.Fatalf("read candidates/0001.json: %v", err)
	}
	if accepted.ProposalKind != proposalReflection || accepted.Prompts["answer"] != "answer v2" {
		t.Fatalf("accepted record = %#v", accepted)
	}

	events, err := readEvents(t, paths.EventsPath)
	if err != nil {
		t.Fatalf("read events: %v", err)
	}
	if !containsEventType(events, eventCandidateAccepted) {
		t.Fatalf("events = %#v, want candidate_accepted", events)
	}
}

func TestOptimizePersistsRejectedProposalEvent(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	runDir := t.TempDir()
	opts := baseOpts(prog, train, engineConfig(20, 2, 10))
	opts.Val = nil
	opts.RunDir = runDir
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{proposal: "answer seed"}

	_, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}

	paths := newRunArtifacts(runDir)
	if _, err := os.Stat(filepath.Join(paths.CandidatesDir, "0001.json")); err == nil {
		t.Fatal("candidates/0001.json exists, want only seed")
	}

	events, err := readEvents(t, paths.EventsPath)
	if err != nil {
		t.Fatalf("read events: %v", err)
	}
	if !containsEventType(events, eventCandidateRejected) {
		t.Fatalf("events = %#v, want candidate_rejected", events)
	}
}

func TestOptimizePersistsProposalFailedEvent(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	runDir := t.TempDir()
	opts := baseOpts(prog, train, engineConfig(20, 2, 11))
	opts.Val = nil
	opts.RunDir = runDir
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{err: errors.New("reflection failed")}

	_, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}

	paths := newRunArtifacts(runDir)
	if _, err := os.Stat(filepath.Join(paths.CandidatesDir, "0001.json")); err == nil {
		t.Fatal("candidates/0001.json exists, want only seed")
	}

	events, err := readEvents(t, paths.EventsPath)
	if err != nil {
		t.Fatalf("read events: %v", err)
	}
	ev, ok := findEventType(events, eventProposalFailed)
	if !ok {
		t.Fatalf("events = %#v, want proposal_failed", events)
	}
	if ev.Reason != "reflection failed" {
		t.Fatalf("proposal_failed reason = %q, want reflection failed", ev.Reason)
	}
}

func TestOptimizeTreatsEmptyReflectionProposalAsProposalFailed(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(4)
	runDir := t.TempDir()
	opts := baseOpts(prog, train, engineConfig(20, 2, 19))
	opts.Val = nil
	opts.RunDir = runDir
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{proposal: "   "}

	_, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}

	paths := newRunArtifacts(runDir)
	if _, err := os.Stat(filepath.Join(paths.CandidatesDir, "0001.json")); err == nil {
		t.Fatal("candidates/0001.json exists, want only seed")
	}

	events, err := readEvents(t, paths.EventsPath)
	if err != nil {
		t.Fatalf("read events: %v", err)
	}
	ev, ok := findEventType(events, eventProposalFailed)
	if !ok {
		t.Fatalf("events = %#v, want proposal_failed", events)
	}
	if ev.Reason != "empty reflected instruction" {
		t.Fatalf("proposal_failed reason = %q, want empty reflected instruction", ev.Reason)
	}
}

func TestOptimizeRejectsEvaluatorResultLengthMismatch(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(3)
	opts := baseOpts(prog, train, engineConfig(20, 2, 21))
	opts.Val = nil
	opts.Evaluator = badLengthEvaluator{}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	_, err := Optimize(context.Background(), opts)
	if err == nil {
		t.Fatal("Optimize() error = nil, want result length mismatch")
	}
	if !errors.Is(err, errEvaluatorResultLength) {
		t.Fatalf("Optimize() error = %v, want errEvaluatorResultLength", err)
	}
}

func TestOptimizeUsesRoundRobinModuleSelection(t *testing.T) {
	prog := twoModuleProgram()
	train := makeTrainExamples(4)
	opts := baseOpts(prog, train, engineConfig(30, 2, 7))
	ref := &scriptedReflector{proposal: "answer v2"}
	trainLen := len(train)
	opts.Evaluator = &scriptedEvaluator{
		trainSize: trainLen,
		scoreFor: func(c Candidate, examples []program.Example) float64 {
			if len(examples) < trainLen {
				if c["answer"] == "answer v2" {
					return 1
				}
				return 0
			}
			return 0
		},
	}
	opts.Reflector = ref

	_, err := Optimize(context.Background(), opts)
	if err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	want := []string{"retriever", "answer"}
	if len(ref.modules) < 2 {
		t.Fatalf("reflector modules = %v, want at least 2 calls", ref.modules)
	}
	if !reflect.DeepEqual(ref.modules[:2], want) {
		t.Fatalf("first reflection modules = %v, want %v", ref.modules[:2], want)
	}
}

func validEngineConfig() config.Config {
	return config.Config{
		Budget:              100,
		MinibatchSize:       3,
		DefaultMaxToolSteps: 8,
		Seed:                42,
		ReflectionModel:     "test/reflection",
		TaskModel:           "test/task",
		Metric:              config.Metric{Kind: "exact_match", Field: "answer"},
	}
}

func engineConfig(budget, minibatch int, seed int64) config.Config {
	cfg := validEngineConfig()
	cfg.Budget = budget
	cfg.MinibatchSize = minibatch
	cfg.Seed = seed
	return cfg
}

func baseOpts(prog program.Program, train []program.Example, cfg config.Config) Options {
	return Options{
		Program: prog,
		Config:  cfg,
		Train:   train,
		Val:     []program.Example{{Input: map[string]any{"question": "vq"}, Expected: map[string]any{"answer": "va"}}},
	}
}

func singleModuleProgram() program.Program {
	return program.Program{
		Modules: []program.Module{{Name: "answer", Prompt: "answer seed"}},
	}
}

func twoModuleProgram() program.Program {
	return program.Program{
		Modules: []program.Module{
			{Name: "retriever", Prompt: "retrieve context"},
			{Name: "answer", Prompt: "answer seed"},
		},
	}
}

func makeTrainExamples(n int) []program.Example {
	out := make([]program.Example, n)
	for i := range out {
		out[i] = program.Example{
			Input:    map[string]any{"question": "q"},
			Expected: map[string]any{"answer": "a"},
		}
	}
	return out
}

type fakeEvaluator struct{}

func (fakeEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return []ExampleResult{{Score: 1, Feedback: "ok"}}, nil
}

type fakeReflector struct{}

func (fakeReflector) Propose(context.Context, ReflectionRequest) (string, error) {
	return "better prompt", nil
}

type scriptedEvaluator struct {
	trainSize int
	scoreFor  func(candidate Candidate, examples []program.Example) float64
}

func (e *scriptedEvaluator) Evaluate(_ context.Context, candidate Candidate, examples []program.Example) ([]ExampleResult, error) {
	scoreFn := e.scoreFor
	if scoreFn == nil {
		scoreFn = func(Candidate, []program.Example) float64 { return 0 }
	}
	out := make([]ExampleResult, len(examples))
	for i := range out {
		out[i] = ExampleResult{Score: scoreFn(candidate, examples), Feedback: "ok"}
	}
	return out, nil
}

type scriptedReflector struct {
	proposal string
	err      error
	modules  []string
}

type badLengthEvaluator struct{}

func (badLengthEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return []ExampleResult{{Score: 1, Feedback: "ok"}}, nil
}

func (r *scriptedReflector) Propose(_ context.Context, req ReflectionRequest) (string, error) {
	r.modules = append(r.modules, req.ModuleName)
	if r.err != nil {
		return "", r.err
	}
	return r.proposal, nil
}

func assertFileExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("Stat(%q) error: %v", path, err)
	}
}

func readEvents(t *testing.T, path string) ([]eventRecord, error) {
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

func containsEventType(events []eventRecord, typ string) bool {
	_, ok := findEventType(events, typ)
	return ok
}

func findEventType(events []eventRecord, typ string) (eventRecord, bool) {
	for _, ev := range events {
		if ev.Type == typ {
			return ev, true
		}
	}
	return eventRecord{}, false
}

func TestOptimizeNoPersistenceWhenRunDirEmpty(t *testing.T) {
	prog := singleModuleProgram()
	train := makeTrainExamples(3)
	runParent := t.TempDir()
	watchDir := filepath.Join(runParent, "should-not-be-created")
	opts := baseOpts(prog, train, engineConfig(3, 1, 12))
	opts.Val = nil
	opts.RunDir = ""
	opts.Evaluator = &scriptedEvaluator{
		trainSize: len(train),
		scoreFor:  func(Candidate, []program.Example) float64 { return 1 },
	}
	opts.Reflector = &scriptedReflector{proposal: "answer v2"}

	if _, err := Optimize(context.Background(), opts); err != nil {
		t.Fatalf("Optimize() unexpected error: %v", err)
	}
	if _, err := os.Stat(watchDir); !os.IsNotExist(err) {
		if err == nil {
			t.Fatalf("%q should not exist when RunDir is empty", watchDir)
		}
		t.Fatalf("Stat(%q) error = %v, want not exist", watchDir, err)
	}
	entries, err := os.ReadDir(runParent)
	if err != nil {
		t.Fatalf("ReadDir(%q): %v", runParent, err)
	}
	if len(entries) != 0 {
		t.Fatalf("run parent has %d entries, want 0; names: %s", len(entries), dirNames(entries))
	}
}

func dirNames(entries []os.DirEntry) string {
	names := make([]string, len(entries))
	for i, e := range entries {
		names[i] = e.Name()
	}
	return strings.Join(names, ", ")
}
