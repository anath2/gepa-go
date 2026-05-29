package gepa

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadProblemHappyPath(t *testing.T) {
	paths := writeProblemFixtures(t)

	problem, err := LoadProblem(ProblemPaths{
		Program: paths["program"],
		Config:  paths["config"],
		Train:   paths["train"],
		Val:     paths["val"],
	})
	if err != nil {
		t.Fatalf("LoadProblem() unexpected error: %v", err)
	}
	if len(problem.Program.Modules) != 1 {
		t.Fatalf("modules = %d, want 1", len(problem.Program.Modules))
	}
	if problem.Config.Budget != 10 || problem.Config.MinibatchSize != 3 {
		t.Fatalf("config = %#v, want budget=10 minibatch=3", problem.Config)
	}
	if len(problem.Train) != 2 || len(problem.Val) != 1 {
		t.Fatalf("train/val lengths = %d/%d, want 2/1", len(problem.Train), len(problem.Val))
	}
}

func TestLoadProblemRejectsBadMetricField(t *testing.T) {
	paths := writeProblemFixtures(t)
	rewriteProblemFixture(t, paths["config"], `{
		"budget": 10, "seed": 0,
		"reflection_model": "x", "task_model": "y",
		"metric": {"kind": "exact_match", "field": "score"}
	}`)

	_, err := LoadProblem(ProblemPaths{
		Program: paths["program"],
		Config:  paths["config"],
		Train:   paths["train"],
		Val:     paths["val"],
	})
	if err == nil {
		t.Fatal("LoadProblem() error = nil, want metric field error")
	}
	want := paths["config"] + `: metric.field "score" not declared in program.json: modules[last].output_schema`
	if err.Error() != want {
		t.Fatalf("LoadProblem() error = %q, want %q", err.Error(), want)
	}
}

func TestLoadProblemMissingProgramFile(t *testing.T) {
	paths := writeProblemFixtures(t)
	missing := filepath.Join(t.TempDir(), "missing-program.json")

	_, err := LoadProblem(ProblemPaths{
		Program: missing,
		Config:  paths["config"],
		Train:   paths["train"],
		Val:     paths["val"],
	})
	if err == nil {
		t.Fatal("LoadProblem() error = nil, want open failure")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("LoadProblem() error = %v, want os.ErrNotExist", err)
	}
}

func writeProblemFixtures(t *testing.T) map[string]string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"program": `{
			"modules": [{
				"name": "answer",
				"prompt": "Answer the question.",
				"input_schema":  {"type":"object","fields":{"question":"string"}},
				"output_schema": {"type":"object","fields":{"answer":"string"}}
			}]
		}`,
		"config": `{
			"budget": 10,
			"seed": 42,
			"reflection_model": "anthropic/claude-3.5-sonnet",
			"task_model": "openai/gpt-4o-mini",
			"metric": {"kind": "exact_match", "field": "answer"}
		}`,
		"train": `{"input":{"question":"q1"},"expected":{"answer":"a1"}}
{"input":{"question":"q2"},"expected":{"answer":"a2"}}
`,
		"val": `{"input":{"question":"vq"},"expected":{"answer":"va"}}
`,
	}
	extByKey := map[string]string{
		"program": ".json",
		"config":  ".json",
		"train":   ".jsonl",
		"val":     ".jsonl",
	}
	paths := make(map[string]string, len(files))
	for key, content := range files {
		path := filepath.Join(dir, key+extByKey[key])
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
		paths[key] = path
	}
	return paths
}

func rewriteProblemFixture(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("rewrite %s: %v", path, err)
	}
}
