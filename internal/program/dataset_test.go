package program

import (
	"os"
	"path/filepath"
	"testing"
)

// firstInputSchema is the input_schema used by all dataset tests:
// {"type":"object","fields":{"question":"string"}}.
func firstInputSchema() Schema {
	return Schema{Kind: KindObject, Fields: map[string]Schema{"question": {Kind: KindString}}}
}

func writeJSONL(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "data.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestDatasetLoadOK(t *testing.T) {
	path := writeJSONL(t, `{"input":{"question":"q1"},"expected":{"answer":"a1"}}
{"input":{"question":"q2"},"expected":{"answer":"a2"}}
`)
	examples, err := LoadDataset(path, firstInputSchema(), "answer")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(examples) != 2 {
		t.Errorf("got %d examples, want 2", len(examples))
	}
}

func TestDatasetBlankLines(t *testing.T) {
	path := writeJSONL(t, `

{"input":{"question":"q1"},"expected":{"answer":"a1"}}

{"input":{"question":"q2"},"expected":{"answer":"a2"}}

`)
	examples, err := LoadDataset(path, firstInputSchema(), "answer")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(examples) != 2 {
		t.Errorf("got %d examples, want 2", len(examples))
	}
}

func TestDatasetTrailingNewlineNoExtraRow(t *testing.T) {
	path := writeJSONL(t, `{"input":{"question":"q"},"expected":{"answer":"a"}}
`)
	examples, err := LoadDataset(path, firstInputSchema(), "answer")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(examples) != 1 {
		t.Errorf("got %d examples, want 1", len(examples))
	}
}

func TestDatasetCRLF(t *testing.T) {
	path := writeJSONL(t, "{\"input\":{\"question\":\"q\"},\"expected\":{\"answer\":\"a\"}}\r\n")
	examples, err := LoadDataset(path, firstInputSchema(), "answer")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(examples) != 1 {
		t.Errorf("got %d examples, want 1", len(examples))
	}
}

func TestDatasetD1InvalidJSON(t *testing.T) {
	path := writeJSONL(t, "{not json}\n")
	_, err := LoadDataset(path, firstInputSchema(), "answer")
	if err == nil {
		t.Fatal("expected error")
	}
	wantPrefix := path + ":1: invalid JSON:"
	if !startsWith(err.Error(), wantPrefix) {
		t.Errorf("got %q, want prefix %q", err.Error(), wantPrefix)
	}
}

func TestDatasetD2UnknownKey(t *testing.T) {
	path := writeJSONL(t, `{"input":{"question":"q"},"expected":{"answer":"a"},"extra":1}`)
	_, err := LoadDataset(path, firstInputSchema(), "answer")
	want := path + `:1: unknown key "extra" (allowed: input, expected)`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestDatasetD3MissingInput(t *testing.T) {
	path := writeJSONL(t, `{"expected":{"answer":"a"}}`)
	_, err := LoadDataset(path, firstInputSchema(), "answer")
	want := path + ":1: input: required object"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestDatasetD4MissingExpected(t *testing.T) {
	path := writeJSONL(t, `{"input":{"question":"q"}}`)
	_, err := LoadDataset(path, firstInputSchema(), "answer")
	want := path + ":1: expected: required object"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestDatasetD5InputTypeMismatch(t *testing.T) {
	path := writeJSONL(t, `{"input":{"question":"ok"},"expected":{"answer":"a"}}
{"input":{"question":42},"expected":{"answer":"a"}}
`)
	_, err := LoadDataset(path, firstInputSchema(), "answer")
	want := path + ":2: input.question: expected string, got float64"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestDatasetD6ExpectedFieldMissing(t *testing.T) {
	path := writeJSONL(t, `{"input":{"question":"q"},"expected":{"other":"x"}}`)
	_, err := LoadDataset(path, firstInputSchema(), "answer")
	want := path + ":1: expected.answer: required"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestDatasetMissingFile(t *testing.T) {
	_, err := LoadDataset("/tmp/does/not/exist.jsonl", firstInputSchema(), "answer")
	if err == nil {
		t.Fatal("expected error")
	}
}

func startsWith(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}
