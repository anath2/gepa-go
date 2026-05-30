package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setupStubLLM routes chat completions to an httptest server for CLI integration tests.
func setupStubLLM(t *testing.T, moduleOutput string) {
	t.Helper()
	t.Setenv("API_KEY", "test-key")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":` + jsonString(moduleOutput) + `}}]}`))
	}))
	t.Cleanup(srv.Close)
	t.Setenv("BASE_URL", srv.URL)
}

func jsonString(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// writeMinimalFixtures writes a complete, minimal-valid set of program /
// config / train / val files under t.TempDir() and returns their paths keyed
// by the corresponding CLI flag name.
func writeMinimalFixtures(t *testing.T) map[string]string {
	t.Helper()
	dir := t.TempDir()
	logDir := filepath.Join(dir, "runs")
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
			"metric": {"kind": "exact_match", "field": "answer"},
			"log_dir": "` + filepath.ToSlash(logDir) + `"
		}`,
		"train": `{"input":{"question":"q1"},"expected":{"answer":"a1"}}
{"input":{"question":"q2"},"expected":{"answer":"a2"}}
`,
		"val": `{"input":{"question":"vq"},"expected":{"answer":"va"}}
`,
	}
	extByKey := map[string]string{
		"program": ".json", "config": ".json",
		"train": ".jsonl", "val": ".jsonl",
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

func TestOptimizeHappyPathSummary(t *testing.T) {
	setupStubLLM(t, `{"answer":"a1"}`)
	p := writeMinimalFixtures(t)
	out, _, err := runCmd(t, "optimize",
		"--program", p["program"],
		"--config", p["config"],
		"--train", p["train"],
		"--val", p["val"],
		"--run-id", "integration-test",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{
		"program:  1 modules, 0 tools",
		`config:   budget=10 minibatch=3 seed=42`,
		`models:   task=openai/gpt-4o-mini  reflection=anthropic/claude-3.5-sonnet`,
		`metric:   exact_match on "answer"`,
		"train:    2 examples",
		"val:      1 examples",
		"run:",
		"best:     candidate",
		"metric_calls=",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing summary line %q\nfull output:\n%s", want, out)
		}
	}
	runDir := filepath.Join(filepath.Dir(p["program"]), "runs", "integration-test")
	if _, err := os.Stat(filepath.Join(runDir, "result.json")); err != nil {
		t.Fatalf("result.json missing under %s: %v", runDir, err)
	}
}

// rewrite is a small helper that takes a fixture map from writeMinimalFixtures
// and overwrites one of the files. Returns the (now-shadowed) path map.
func rewrite(t *testing.T, paths map[string]string, key, content string) {
	t.Helper()
	if err := os.WriteFile(paths[key], []byte(content), 0o644); err != nil {
		t.Fatalf("rewrite %s: %v", key, err)
	}
}

func TestOptimizeProgramEmptyModules(t *testing.T) {
	paths := writeMinimalFixtures(t)
	rewrite(t, paths, "program", `{"modules": []}`)
	_, _, err := runCmd(t, "optimize",
		"--program", paths["program"], "--config", paths["config"],
		"--train", paths["train"], "--val", paths["val"])
	if err == nil {
		t.Fatal("expected error")
	}
	want := paths["program"] + ": modules: at least one module required"
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}

func TestOptimizeWrongMetricField(t *testing.T) {
	paths := writeMinimalFixtures(t)
	rewrite(t, paths, "config", `{
		"budget": 10, "seed": 0,
		"reflection_model": "x", "task_model": "y",
		"metric": {"kind": "exact_match", "field": "score"}
	}`)
	_, _, err := runCmd(t, "optimize",
		"--program", paths["program"], "--config", paths["config"],
		"--train", paths["train"], "--val", paths["val"])
	if err == nil {
		t.Fatal("expected error")
	}
	want := paths["config"] + `: metric.field "score" not declared in program.json: modules[last].output_schema`
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}

func TestOptimizeDatasetTypeMismatch(t *testing.T) {
	paths := writeMinimalFixtures(t)
	rewrite(t, paths, "train", `{"input":{"question":"ok"},"expected":{"answer":"a"}}
{"input":{"question":42},"expected":{"answer":"a"}}
`)
	_, _, err := runCmd(t, "optimize",
		"--program", paths["program"], "--config", paths["config"],
		"--train", paths["train"], "--val", paths["val"])
	if err == nil {
		t.Fatal("expected error")
	}
	want := paths["train"] + ":2: input.question: expected string, got float64"
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}

func TestOptimizeUnknownDatasetKey(t *testing.T) {
	paths := writeMinimalFixtures(t)
	rewrite(t, paths, "val", `{"input":{"question":"q"},"expected":{"answer":"a"},"extra":1}
`)
	_, _, err := runCmd(t, "optimize",
		"--program", paths["program"], "--config", paths["config"],
		"--train", paths["train"], "--val", paths["val"])
	if err == nil {
		t.Fatal("expected error")
	}
	want := paths["val"] + `:1: unknown key "extra" (allowed: input, expected)`
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}

func TestOptimizeMissingExpectedFieldInRow(t *testing.T) {
	paths := writeMinimalFixtures(t)
	rewrite(t, paths, "train", `{"input":{"question":"q"},"expected":{"other":"x"}}
`)
	_, _, err := runCmd(t, "optimize",
		"--program", paths["program"], "--config", paths["config"],
		"--train", paths["train"], "--val", paths["val"])
	if err == nil {
		t.Fatal("expected error")
	}
	want := paths["train"] + ":1: expected.answer: required"
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}
