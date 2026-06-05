package rollout

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/program"
)

func externalEvaluatorCommand(response string) []string {
	return []string{"sh", "-c", fmt.Sprintf("cat >/dev/null; printf '%%s\\n' %q", response)}
}

func captureStdinCommand(capturePath, response string) []string {
	return []string{"sh", "-c", fmt.Sprintf("cat > %q; printf '%%s\\n' %q", capturePath, response)}
}

func TestRunExternalEvaluatorReturnsScoreAndFeedback(t *testing.T) {
	got, err := runExternalEvaluator(context.Background(), program.ModuleEvaluator{
		Kind:    "external",
		Command: externalEvaluatorCommand(`{"score":0.75,"feedback":"module feedback"}`),
	}, "mod", map[string]any{"x": "in"}, map[string]any{"y": "out"}, program.Example{
		Input:    map[string]any{"x": "in"},
		Expected: map[string]any{"answer": "want"},
	})
	if err != nil {
		t.Fatalf("runExternalEvaluator() error = %v", err)
	}
	if got.Source != gepa.EvalSourceExternalEvaluator {
		t.Fatalf("Source = %q, want external_evaluator", got.Source)
	}
	if got.Score != 0.75 {
		t.Fatalf("Score = %v, want 0.75", got.Score)
	}
	if got.Feedback != "module feedback" {
		t.Fatalf("Feedback = %q", got.Feedback)
	}
}

func TestRunExternalEvaluatorReceivesStdinPayload(t *testing.T) {
	capturePath := filepath.Join(t.TempDir(), "stdin.json")
	_, err := runExternalEvaluator(context.Background(), program.ModuleEvaluator{
		Kind:    "external",
		Command: captureStdinCommand(capturePath, `{"score":1,"feedback":"ok"}`),
	}, "mod", map[string]any{"x": "in"}, map[string]any{"y": "out"}, program.Example{
		Input:    map[string]any{"x": "in"},
		Expected: map[string]any{"answer": "want"},
	})
	if err != nil {
		t.Fatalf("runExternalEvaluator() error = %v", err)
	}
	data, err := os.ReadFile(capturePath)
	if err != nil {
		t.Fatalf("ReadFile(capture) error = %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("Unmarshal payload error = %v", err)
	}
	if payload["module_name"] != "mod" {
		t.Fatalf("module_name = %v, want mod", payload["module_name"])
	}
	example, ok := payload["example"].(map[string]any)
	if !ok {
		t.Fatalf("example = %#v, want object", payload["example"])
	}
	if _, ok := example["input"]; !ok {
		t.Fatal("payload.example.input missing")
	}
	if _, ok := example["expected"]; !ok {
		t.Fatal("payload.example.expected missing")
	}
}

func TestRunExternalEvaluatorNonzeroExitIsSoftFailure(t *testing.T) {
	got, err := runExternalEvaluator(context.Background(), program.ModuleEvaluator{
		Kind:    "external",
		Command: []string{"sh", "-c", "echo bad >&2; exit 2"},
	}, "mod", map[string]any{"x": "in"}, map[string]any{"y": "out"}, program.Example{})
	if err != nil {
		t.Fatalf("runExternalEvaluator() error = %v, want nil soft failure", err)
	}
	if got.Score != 0 {
		t.Fatalf("Score = %v, want 0 on nonzero exit", got.Score)
	}
	if !strings.Contains(got.Feedback, "bad") {
		t.Fatalf("Feedback = %q, want stderr content", got.Feedback)
	}
}

func TestRunExternalEvaluatorRejectsInvalidStdout(t *testing.T) {
	_, err := runExternalEvaluator(context.Background(), program.ModuleEvaluator{
		Kind:    "external",
		Command: []string{"sh", "-c", "echo not-json"},
	}, "mod", nil, map[string]any{"y": "out"}, program.Example{})
	if err == nil {
		t.Fatal("runExternalEvaluator() error = nil, want invalid stdout error")
	}
}

func TestEvaluatorExternalEvaluatorWiring(t *testing.T) {
	prog := program.Program{
		Modules: []program.Module{{
			Name:         "mod",
			InputSchema:  objectSchema("x", program.KindString),
			OutputSchema: objectSchema("y", program.KindString),
			Evaluator: &program.ModuleEvaluator{
				Kind:    "external",
				Command: externalEvaluatorCommand(`{"score":0.75,"feedback":"module feedback"}`),
			},
		}},
	}
	eval := Evaluator{
		Program: prog,
		Config:  config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "y"}},
		Model:   &fakeTaskModel{responses: []ModuleResponse{{Output: map[string]any{"y": "out"}}}},
	}

	results, err := eval.Evaluate(context.Background(), gepa.Candidate{"mod": "prompt"}, []program.Example{
		{Input: map[string]any{"x": "in"}, Expected: map[string]any{"y": "want"}},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	trace := results[0].ModuleTraces[0]
	if trace.Evaluation == nil || trace.Evaluation.Score != 0.75 {
		t.Fatalf("module evaluation = %#v, want score 0.75", trace.Evaluation)
	}
	if results[0].Score != 0 {
		t.Fatalf("global Score = %v, want 0 (final output mismatch)", results[0].Score)
	}
}
