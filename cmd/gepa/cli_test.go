package main

import (
	"bytes"
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

// TestOptimizeValidPaths covers the CLI-level path-wiring concern: that all
// four required flags resolve to readable files and the body runs successfully.
// End-to-end summary-content assertions live in optimize_integration_test.go.
func TestOptimizeValidPaths(t *testing.T) {
	setupStubLLM(t, `{"answer":"a1"}`)
	paths := writeMinimalFixtures(t)

	out, _, err := runCmd(t, "optimize",
		"--program", paths["program"],
		"--config", paths["config"],
		"--train", paths["train"],
		"--val", paths["val"],
		"--run-id", "valid-paths",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "program:") || !strings.Contains(out, "train:") || !strings.Contains(out, "best:") {
		t.Errorf("expected summary block on stdout, got:\n%s", out)
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
	want := "resume: " + dir
	if !strings.Contains(out, want) {
		t.Errorf("stdout = %q, want substring %q", out, want)
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
