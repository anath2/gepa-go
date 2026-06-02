package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anath2/gepa-go/internal/program"
)

func validConfig() Config {
	return Config{
		Budget:              100,
		MinibatchSize:       3,
		DefaultMaxToolSteps: 8,
		Seed:                42,
		ReflectionModel:     "anthropic/claude-3.5-sonnet",
		TaskModel:           "openai/gpt-4o-mini",
		Metric:              Metric{Kind: "exact_match", Field: "answer"},
		LogDir:              "./runs/",
	}
}

func validProgram() program.Program {
	return program.Program{
		Modules: []program.Module{
			{
				Name:         "answer",
				InputSchema:  program.Schema{Kind: program.KindObject, Fields: map[string]program.Schema{"q": {Kind: program.KindString}}},
				OutputSchema: program.Schema{Kind: program.KindObject, Fields: map[string]program.Schema{"answer": {Kind: program.KindString}}},
			},
		},
	}
}

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadConfigOK(t *testing.T) {
	path := writeConfig(t, `{
        "budget": 200, "seed": 42,
        "reflection_model": "x", "task_model": "y",
        "metric": {"kind": "exact_match", "field": "answer"}
    }`)
	c, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.MinibatchSize != 3 {
		t.Errorf("MinibatchSize default not applied: %d", c.MinibatchSize)
	}
	if c.DefaultMaxToolSteps != 8 {
		t.Errorf("DefaultMaxToolSteps default not applied: %d", c.DefaultMaxToolSteps)
	}
	if c.LogDir != "./runs/" {
		t.Errorf("LogDir default not applied: %q", c.LogDir)
	}
}

func TestLoadConfigReasoningEffortPerModel(t *testing.T) {
	path := writeConfig(t, `{
		"budget": 200, "seed": 42,
		"reflection_model": "x", "task_model": "y",
		"reflection_reasoning_effort": "none",
		"task_reasoning_effort": "minimal",
		"metric": {"kind": "exact_match", "field": "answer"}
	}`)
	c, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.ReflectionReasoningEffort != "none" {
		t.Fatalf("ReflectionReasoningEffort = %q, want none", c.ReflectionReasoningEffort)
	}
	if c.TaskReasoningEffort != "minimal" {
		t.Fatalf("TaskReasoningEffort = %q, want minimal", c.TaskReasoningEffort)
	}
}

func TestValidateRejectsInvalidReasoningEffort(t *testing.T) {
	c := validConfig()
	c.TaskReasoningEffort = "disabled"
	err := c.validate()
	want := `config.json: task_reasoning_effort: must be one of "xhigh", "high", "medium", "low", "minimal", "none", got "disabled"`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestLoadConfigUnknownField(t *testing.T) {
	path := writeConfig(t, `{"budget":1,"surprise":2}`)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestValidateC1ZeroBudget(t *testing.T) {
	c := validConfig()
	c.Budget = 0
	err := c.validate()
	want := "config.json: budget: must be > 0"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateC2ZeroMinibatch(t *testing.T) {
	c := validConfig()
	c.MinibatchSize = 0
	// Note: applyDefaults is not auto-called by Validate(). Call manually
	// to skip the default — simulating a user explicitly setting it to 0
	// after defaults are applied is not possible through Load(), but Validate
	// must defend against it.
	c.MinibatchSize = -1
	err := c.validate()
	want := "config.json: minibatch_size: must be > 0"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateC3ZeroMaxToolSteps(t *testing.T) {
	c := validConfig()
	c.DefaultMaxToolSteps = -1
	err := c.validate()
	want := "config.json: default_max_tool_steps: must be > 0"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateC4EmptyReflectionModel(t *testing.T) {
	c := validConfig()
	c.ReflectionModel = ""
	err := c.validate()
	want := "config.json: reflection_model: required"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateC5EmptyTaskModel(t *testing.T) {
	c := validConfig()
	c.TaskModel = ""
	err := c.validate()
	want := "config.json: task_model: required"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateC6UnsupportedMetricKind(t *testing.T) {
	c := validConfig()
	c.Metric.Kind = "f1"
	err := c.validate()
	want := `config.json: metric.kind: only "exact_match" supported in v0, got "f1"`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateC7EmptyMetricField(t *testing.T) {
	c := validConfig()
	c.Metric.Field = ""
	err := c.validate()
	want := "config.json: metric.field: required"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateAgainstProgramOK(t *testing.T) {
	c := validConfig()
	p := validProgram()
	if err := c.ValidateAgainstProgram(p); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidateAgainstProgramMissingField(t *testing.T) {
	c := validConfig()
	c.Metric.Field = "score"
	p := validProgram()
	err := c.ValidateAgainstProgram(p)
	want := `config.json: metric.field "score" not declared in program.json: modules[last].output_schema`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateAgainstProgramWrongKind(t *testing.T) {
	c := validConfig()
	p := validProgram()
	// Replace the output field type with int — metric expects string.
	last := &p.Modules[len(p.Modules)-1]
	last.OutputSchema = program.Schema{Kind: program.KindObject, Fields: map[string]program.Schema{
		"answer": {Kind: program.KindInt},
	}}
	err := c.ValidateAgainstProgram(p)
	want := `config.json: metric.field "answer" must be type string in program.json: modules[last].output_schema.answer, got int`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}
