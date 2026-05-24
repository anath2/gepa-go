package gepa

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/program"
)

func TestOptionsHoldsInputsAndDependencies(t *testing.T) {
	prog := program.Program{Modules: []program.Module{{Name: "answer", Prompt: "answer"}}}
	cfg := config.Config{Budget: 10, MinibatchSize: 3, Seed: 7}
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
		Config:  config.Config{Budget: 10, MinibatchSize: 3, Seed: 7},
		Train:   []program.Example{{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "a"}}},
	}

	_, err := Optimize(context.Background(), opts)
	if err == nil {
		t.Fatal("Optimize() error = nil, want not implemented")
	}
	if !strings.Contains(err.Error(), "gepa optimize loop not implemented") {
		t.Fatalf("Optimize() error = %q, want not implemented message", err.Error())
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
	if !errors.Is(err, ErrEvaluatorNotImplemented) {
		t.Fatalf("Evaluate() error = %v, want ErrEvaluatorNotImplemented", err)
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
	if !errors.Is(err, ErrReflectorNotImplemented) {
		t.Fatalf("Propose() error = %v, want ErrReflectorNotImplemented", err)
	}
}

type fakeEvaluator struct{}

func (fakeEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return []ExampleResult{{Score: 1, Feedback: "ok"}}, nil
}

type fakeReflector struct{}

func (fakeReflector) Propose(context.Context, ReflectionRequest) (string, error) {
	return "better prompt", nil
}
