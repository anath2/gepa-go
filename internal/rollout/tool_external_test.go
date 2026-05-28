package rollout

import (
	"context"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/program"
)

func TestRunExternalToolSuccess(t *testing.T) {
	tool := program.Tool{
		Name:         "echo",
		Command:      []string{"sh", "-c", "cat"},
		OutputSchema: objectSchema("text", program.KindString),
	}

	got, err := runExternalTool(context.Background(), tool, map[string]any{"text": "hello"})
	if err != nil {
		t.Fatalf("runExternalTool() error = %v", err)
	}
	if got["text"] != "hello" {
		t.Fatalf("output.text = %#v, want %#v", got["text"], "hello")
	}
}

func TestRunExternalToolNonZeroExit(t *testing.T) {
	tool := program.Tool{
		Name:         "boom",
		Command:      []string{"sh", "-c", "echo bad 1>&2; exit 3"},
		OutputSchema: objectSchema("text", program.KindString),
	}

	_, err := runExternalTool(context.Background(), tool, map[string]any{"text": "hello"})
	if err == nil {
		t.Fatal("runExternalTool() error = nil, want non-zero exit error")
	}
	if !strings.Contains(err.Error(), "failed") {
		t.Fatalf("error = %q, want contains %q", err.Error(), "failed")
	}
}

func TestRunExternalToolInvalidJSON(t *testing.T) {
	tool := program.Tool{
		Name:         "badjson",
		Command:      []string{"sh", "-c", "echo not-json"},
		OutputSchema: objectSchema("text", program.KindString),
	}

	_, err := runExternalTool(context.Background(), tool, map[string]any{"text": "hello"})
	if err == nil {
		t.Fatal("runExternalTool() error = nil, want invalid json error")
	}
	if !strings.Contains(err.Error(), "invalid json") {
		t.Fatalf("error = %q, want contains %q", err.Error(), "invalid json")
	}
}
