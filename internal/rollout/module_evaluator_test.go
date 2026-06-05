package rollout

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/program"
)

func TestDecodeFailureEvaluation(t *testing.T) {
	err := fmt.Errorf("%w: unexpected end of JSON input", errDecodeModuleOutput)
	eval := decodeFailureEvaluation("answer", err)
	if eval.Source != gepa.EvalSourceDecode {
		t.Fatalf("Source = %q, want %q", eval.Source, gepa.EvalSourceDecode)
	}
	if eval.Score != 0 {
		t.Fatalf("Score = %v, want 0", eval.Score)
	}
	if eval.Feedback == "" || eval.Error == "" {
		t.Fatalf("Feedback/Error = %q/%q, want non-empty", eval.Feedback, eval.Error)
	}
}

func TestSchemaFailureEvaluation(t *testing.T) {
	err := errors.New("output.answer: expected string, got float64")
	eval := schemaFailureEvaluation("answer", err)
	if eval.Source != gepa.EvalSourceSchema {
		t.Fatalf("Source = %q, want %q", eval.Source, gepa.EvalSourceSchema)
	}
	if eval.Score != 0 {
		t.Fatalf("Score = %v, want 0", eval.Score)
	}
}

func TestRunModuleEvaluatorSkipsWithoutDecl(t *testing.T) {
	module := program.Module{
		Name:         "answer",
		InputSchema:  objectSchema("question", program.KindString),
		OutputSchema: objectSchema("answer", program.KindString),
	}
	got, err := runModuleEvaluator(context.Background(), module, map[string]any{"question": "q"}, map[string]any{"answer": "a"}, program.Example{
		Input:    map[string]any{"question": "q"},
		Expected: map[string]any{"answer": "a"},
	})
	if err != nil {
		t.Fatalf("runModuleEvaluator() error = %v, want nil", err)
	}
	if got != nil {
		t.Fatalf("runModuleEvaluator() = %#v, want nil", got)
	}
}

func TestEvaluatorDecodeFailureAttachesModuleEvaluation(t *testing.T) {
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
	trace := results[0].ModuleTraces[0]
	if trace.Evaluation == nil {
		t.Fatal("ModuleTraces[0].Evaluation = nil, want decode evaluation")
	}
	if trace.Evaluation.Source != gepa.EvalSourceDecode {
		t.Fatalf("Evaluation.Source = %q, want %q", trace.Evaluation.Source, gepa.EvalSourceDecode)
	}
	if trace.Evaluation.Score != 0 {
		t.Fatalf("Evaluation.Score = %v, want 0", trace.Evaluation.Score)
	}
	if trace.Evaluation.Feedback == "" {
		t.Fatal("Evaluation.Feedback = empty, want decode feedback")
	}
}

func TestEvaluatorSchemaFailureAttachesModuleEvaluation(t *testing.T) {
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
	trace := results[0].ModuleTraces[0]
	if trace.Evaluation == nil {
		t.Fatal("ModuleTraces[0].Evaluation = nil, want schema evaluation")
	}
	if trace.Evaluation.Source != gepa.EvalSourceSchema {
		t.Fatalf("Evaluation.Source = %q, want %q", trace.Evaluation.Source, gepa.EvalSourceSchema)
	}
	if trace.Evaluation.Score != 0 {
		t.Fatalf("Evaluation.Score = %v, want 0", trace.Evaluation.Score)
	}
}
