package rollout

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/program"
)

func TestEvaluatorSingleModuleUsesCandidatePromptAndInput(t *testing.T) {
	model := &fakeTaskModel{
		responses: []ModuleResponse{
			{Output: map[string]any{"answer": "Paris"}},
		},
	}
	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{
				{
					Name:         "answer",
					InputSchema:  objectSchema("question", program.KindString),
					OutputSchema: objectSchema("answer", program.KindString),
				},
			},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  model,
	}

	results, err := eval.Evaluate(context.Background(), gepa.Candidate{"answer": "prompt v1"}, []program.Example{
		{Input: map[string]any{"question": "capital?"}, Expected: map[string]any{"answer": "Paris"}},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].Score != 1 {
		t.Fatalf("Score = %v, want 1", results[0].Score)
	}
	if len(model.requests) != 1 {
		t.Fatalf("len(requests) = %d, want 1", len(model.requests))
	}
	req := model.requests[0]
	if req.Instruction != "prompt v1" {
		t.Fatalf("Instruction = %q, want %q", req.Instruction, "prompt v1")
	}
	if got := req.Input["question"]; got != "capital?" {
		t.Fatalf("Input.question = %#v, want %#v", got, "capital?")
	}
}

func TestEvaluatorMultiModuleAccumulatesState(t *testing.T) {
	model := &fakeTaskModel{
		responses: []ModuleResponse{
			{Output: map[string]any{"docs": "from retriever"}},
			{Output: map[string]any{"answer": "final answer"}},
		},
	}
	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{
				{
					Name:         "retriever",
					InputSchema:  objectSchema("question", program.KindString),
					OutputSchema: objectSchema("docs", program.KindString),
				},
				{
					Name: "answerer",
					InputSchema: program.Schema{
						Kind: program.KindObject,
						Fields: map[string]program.Schema{
							"question": {Kind: program.KindString},
							"docs":     {Kind: program.KindString},
						},
					},
					OutputSchema: objectSchema("answer", program.KindString),
				},
			},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  model,
	}

	results, err := eval.Evaluate(context.Background(), gepa.Candidate{
		"retriever": "retrieve",
		"answerer":  "answer",
	}, []program.Example{
		{Input: map[string]any{"question": "q1"}, Expected: map[string]any{"answer": "final answer"}},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if len(model.requests) != 2 {
		t.Fatalf("len(requests) = %d, want 2", len(model.requests))
	}
	second := model.requests[1]
	if got := second.Input["docs"]; got != "from retriever" {
		t.Fatalf("second Input.docs = %#v, want %#v", got, "from retriever")
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	traces := results[0].ModuleTraces
	if len(traces) != 2 {
		t.Fatalf("len(ModuleTraces) = %d, want 2", len(traces))
	}
	if traces[0].ModuleName != "retriever" {
		t.Fatalf("first trace module = %q, want retriever", traces[0].ModuleName)
	}
	if traces[0].Input["question"] != "q1" {
		t.Fatalf("first trace input question = %#v, want q1", traces[0].Input["question"])
	}
	if traces[0].Output["docs"] != "from retriever" {
		t.Fatalf("first trace output docs = %#v, want from retriever", traces[0].Output["docs"])
	}
	if traces[1].ModuleName != "answerer" {
		t.Fatalf("second trace module = %q, want answerer", traces[1].ModuleName)
	}
	if traces[1].Input["docs"] != "from retriever" {
		t.Fatalf("second trace input docs = %#v, want from retriever", traces[1].Input["docs"])
	}
	if traces[1].Output["answer"] != "final answer" {
		t.Fatalf("second trace output answer = %#v, want final answer", traces[1].Output["answer"])
	}
}

func TestEvaluatorMissingPromptReturnsError(t *testing.T) {
	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{{
				Name:         "answer",
				InputSchema:  objectSchema("question", program.KindString),
				OutputSchema: objectSchema("answer", program.KindString),
			}},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  &fakeTaskModel{responses: []ModuleResponse{{Output: map[string]any{"answer": "x"}}}},
	}

	_, err := eval.Evaluate(context.Background(), gepa.Candidate{}, []program.Example{
		{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "x"}},
	})
	if err == nil {
		t.Fatal("Evaluate() error = nil, want malformed-candidate error")
	}
}

func TestEvaluatorBlankPromptReturnsError(t *testing.T) {
	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{{
				Name:         "answer",
				InputSchema:  objectSchema("question", program.KindString),
				OutputSchema: objectSchema("answer", program.KindString),
			}},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  &fakeTaskModel{responses: []ModuleResponse{{Output: map[string]any{"answer": "x"}}}},
	}

	_, err := eval.Evaluate(context.Background(), gepa.Candidate{"answer": "   "}, []program.Example{
		{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "x"}},
	})
	if err == nil {
		t.Fatal("Evaluate() error = nil, want malformed-candidate error")
	}
}

func TestEvaluatorModelFailureReturnsError(t *testing.T) {
	boom := errors.New("model down")
	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{{
				Name:         "answer",
				InputSchema:  objectSchema("question", program.KindString),
				OutputSchema: objectSchema("answer", program.KindString),
			}},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  &fakeTaskModel{err: boom},
	}

	_, err := eval.Evaluate(context.Background(), gepa.Candidate{"answer": "prompt"}, []program.Example{
		{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "x"}},
	})
	if !errors.Is(err, boom) {
		t.Fatalf("Evaluate() error = %v, want %v", err, boom)
	}
}

func TestEvaluatorDecodeFailureReturnsScoredExampleFailure(t *testing.T) {
	decodeErr := fmt.Errorf("%w: unexpected end of JSON input", errDecodeModuleOutput)
	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{{
				Name:         "answer",
				InputSchema:  objectSchema("question", program.KindString),
				OutputSchema: objectSchema("answer", program.KindString),
			}},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  &fakeTaskModel{err: decodeErr},
	}

	results, err := eval.Evaluate(context.Background(), gepa.Candidate{"answer": "prompt"}, []program.Example{
		{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "x"}},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v, want nil", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].Score != 0 {
		t.Fatalf("Score = %v, want 0", results[0].Score)
	}
	if results[0].Feedback == "" {
		t.Fatal("Feedback = empty, want decode failure feedback")
	}
	if !strings.Contains(results[0].Error, decodeErr.Error()) {
		t.Fatalf("Error = %q, want decode failure error", results[0].Error)
	}
}

func TestEvaluatorSchemaValidationFailureReturnsScoredExampleFailure(t *testing.T) {
	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{{
				Name:         "answer",
				InputSchema:  objectSchema("question", program.KindString),
				OutputSchema: objectSchema("answer", program.KindString),
			}},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  &fakeTaskModel{responses: []ModuleResponse{{Output: map[string]any{"answer": 7.0}}}},
	}

	results, err := eval.Evaluate(context.Background(), gepa.Candidate{"answer": "prompt"}, []program.Example{
		{Input: map[string]any{"question": "q"}, Expected: map[string]any{"answer": "x"}},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v, want nil", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].Score != 0 {
		t.Fatalf("Score = %v, want 0", results[0].Score)
	}
	if results[0].Error == "" {
		t.Fatal("Error = empty, want schema validation error")
	}
}

type fakeTaskModel struct {
	requests   []ModuleRequest
	responses  []ModuleResponse
	err        error
	nextResult int
}

func (f *fakeTaskModel) Generate(_ context.Context, req ModuleRequest) (ModuleResponse, error) {
	f.requests = append(f.requests, req)
	if f.err != nil {
		return ModuleResponse{}, f.err
	}
	if f.nextResult >= len(f.responses) {
		return ModuleResponse{Output: map[string]any{}}, nil
	}
	out := f.responses[f.nextResult]
	f.nextResult++
	return out, nil
}

func objectSchema(name string, kind program.Kind) program.Schema {
	return program.Schema{
		Kind: program.KindObject,
		Fields: map[string]program.Schema{
			name: {Kind: kind},
		},
	}
}
