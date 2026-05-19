package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runCmd executes a fresh root command with the given args, capturing stdout
// and stderr separately. cobra writes RunE errors to stderr via SetErr; the
// returned error is the same value.
func runCmd(t *testing.T, args ...string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := newRootCmd()
	var so, se bytes.Buffer
	cmd.SetOut(&so)
	cmd.SetErr(&se)
	cmd.SetArgs(args)
	err = cmd.Execute()
	return so.String(), se.String(), err
}

func TestRootHelp(t *testing.T) {
	out, _, err := runCmd(t, "--help")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{"optimize", "inspect"} {
		if !strings.Contains(out, want) {
			t.Errorf("--help output missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestRootVersion(t *testing.T) {
	out, _, err := runCmd(t, "--version")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, Version) {
		t.Errorf("expected version %q in output, got: %s", Version, out)
	}
}

func TestOptimizeHelp(t *testing.T) {
	out, _, err := runCmd(t, "optimize", "--help")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	wantFlags := []string{
		"--program", "-p",
		"--config", "-c",
		"--train",
		"--val",
		"--run-id",
		"--resume",
		"--log-traces",
	}
	for _, want := range wantFlags {
		if !strings.Contains(out, want) {
			t.Errorf("optimize --help missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestInspectHelp(t *testing.T) {
	out, _, err := runCmd(t, "inspect", "--help")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{"<run-dir>", "--format", "--show-tree", "--show-events"} {
		if !strings.Contains(out, want) {
			t.Errorf("inspect --help missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestOptimizeMissingRequiredFlags(t *testing.T) {
	_, _, err := runCmd(t, "optimize")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	want := "required flag(s) --program, --config, --train, --val not set"
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}

func TestOptimizeMissingFile(t *testing.T) {
	_, _, err := runCmd(t, "optimize",
		"--program", "no.json",
		"--config", "no.json",
		"--train", "no.jsonl",
		"--val", "no.jsonl",
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	want := "program.json not found at no.json"
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}

func TestOptimizeValidPaths(t *testing.T) {
	dir := t.TempDir()
	paths := map[string]string{
		"program": filepath.Join(dir, "p.json"),
		"config":  filepath.Join(dir, "c.json"),
		"train":   filepath.Join(dir, "t.jsonl"),
		"val":     filepath.Join(dir, "v.jsonl"),
	}
	for _, p := range paths {
		if err := os.WriteFile(p, []byte{}, 0o644); err != nil {
			t.Fatalf("touch %s: %v", p, err)
		}
	}

	out, _, err := runCmd(t, "optimize",
		"--program", paths["program"],
		"--config", paths["config"],
		"--train", paths["train"],
		"--val", paths["val"],
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var summary map[string]any
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("stdout not JSON: %v\n%s", err, out)
	}
	if summary["cmd"] != "optimize" {
		t.Errorf("cmd = %v, want optimize", summary["cmd"])
	}
	for key, p := range paths {
		if summary[key] != p {
			t.Errorf("%s = %v, want %s", key, summary[key], p)
		}
	}
	if summary["log_traces"] != false {
		t.Errorf("log_traces = %v, want false", summary["log_traces"])
	}
	if summary["resume"] != "" {
		t.Errorf("resume = %v, want empty", summary["resume"])
	}
}

func TestOptimizeResumeMissing(t *testing.T) {
	bad := filepath.Join(t.TempDir(), "does-not-exist")
	_, _, err := runCmd(t, "optimize", "--resume", bad)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	want := "run directory not found at " + bad
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}

func TestOptimizeResumeExisting(t *testing.T) {
	dir := t.TempDir()
	out, _, err := runCmd(t, "optimize", "--resume", dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("stdout not JSON: %v\n%s", err, out)
	}
	if summary["resume"] != dir {
		t.Errorf("resume = %v, want %s", summary["resume"], dir)
	}
}

func TestInspectNoArg(t *testing.T) {
	_, _, err := runCmd(t, "inspect")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	want := "accepts 1 arg(s), received 0"
	if !strings.Contains(err.Error(), want) {
		t.Errorf("got %q\nwant substring %q", err.Error(), want)
	}
}

func TestInspectExistingDir(t *testing.T) {
	dir := t.TempDir()
	out, _, err := runCmd(t, "inspect", dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "would inspect " + dir + " (format=text show_tree=true show_events=true)"
	if !strings.Contains(out, want) {
		t.Errorf("got %q\nwant substring %q", out, want)
	}
}

func TestInspectBadFormat(t *testing.T) {
	dir := t.TempDir()
	_, _, err := runCmd(t, "inspect", dir, "--format", "yaml")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	want := `invalid --format "yaml": expected one of text, json`
	if err.Error() != want {
		t.Errorf("got %q\nwant %q", err.Error(), want)
	}
}
