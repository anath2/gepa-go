package program

import (
	"os"
	"strings"
	"testing"
)

func programWithEvaluator(evaluator *ModuleEvaluator) Program {
	p := validProgram()
	if evaluator != nil {
		p.Modules[0].Evaluator = evaluator
	}
	return p
}

func TestValidateModuleEvaluatorExternalOK(t *testing.T) {
	p := programWithEvaluator(&ModuleEvaluator{
		Kind:    "external",
		Command: []string{"python", "eval.py"},
	})
	if err := p.validate(); err != nil {
		t.Fatalf("validate() error = %v, want nil", err)
	}
}

func TestValidateModuleEvaluatorUnknownKind(t *testing.T) {
	p := programWithEvaluator(&ModuleEvaluator{
		Kind:    "builtin",
		Command: []string{"python", "eval.py"},
	})
	err := p.validate()
	want := `program.json: modules[0].evaluator.kind: only "external" supported, got "builtin"`
	if err == nil || err.Error() != want {
		t.Fatalf("validate() error = %v, want %q", err, want)
	}
}

func TestValidateModuleEvaluatorEmptyCommand(t *testing.T) {
	p := programWithEvaluator(&ModuleEvaluator{
		Kind:    "external",
		Command: nil,
	})
	err := p.validate()
	want := `program.json: modules[0].evaluator.command: required, non-empty`
	if err == nil || err.Error() != want {
		t.Fatalf("validate() error = %v, want %q", err, want)
	}
}

func TestLoadProgramWithEvaluatorOK(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/program.json"
	body := `{
  "modules": [{
    "name": "answer",
    "prompt": "Answer the question.",
    "input_schema": {"type":"object","fields":{"question":"string"}},
    "output_schema": {"type":"object","fields":{"answer":"string"}},
    "evaluator": {"kind":"external","command":["python","eval.py"]}
  }]
}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if p.Modules[0].Evaluator == nil {
		t.Fatal("Evaluator = nil, want external evaluator")
	}
	if p.Modules[0].Evaluator.Kind != "external" {
		t.Fatalf("Evaluator.Kind = %q, want external", p.Modules[0].Evaluator.Kind)
	}
}

func TestLoadProgramEvaluatorUnknownField(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/program.json"
	body := `{
  "modules": [{
    "name": "answer",
    "prompt": "Answer the question.",
    "input_schema": {"type":"object","fields":{"question":"string"}},
    "output_schema": {"type":"object","fields":{"answer":"string"}},
    "evaluator": {"kind":"external","command":["python","eval.py"],"surprise":1}
  }]
}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("Load() error = nil, want unknown field error")
	}
	if !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("Load() error = %v, want unknown field", err)
	}
}
