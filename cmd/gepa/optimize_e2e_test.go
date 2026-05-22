package main

import (
	"flag"
	"path/filepath"
	"strings"
	"testing"
)

var runE2E = flag.Bool("run-e2e", false, "run high-latency end-to-end tests")

func TestOptimizeChineseNERProblemDefinitionE2E(t *testing.T) {
	if !*runE2E {
		t.Skip("set -run-e2e to run high-latency end-to-end tests")
	}

	fixtureDir := filepath.Join("..", "..", "internal", "testdata", "chinese_ner")
	out, _, err := runCmd(t, "optimize",
		"--program", filepath.Join(fixtureDir, "program.json"),
		"--config", filepath.Join(fixtureDir, "config.json"),
		"--train", filepath.Join(fixtureDir, "train.jsonl"),
		"--val", filepath.Join(fixtureDir, "val.jsonl"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{
		"program:  3 modules, 0 tools",
		`config:   budget=12 minibatch=3 seed=20260522`,
		`models:   task=arcee-ai/trinity-mini  reflection=~moonshotai/kimi-latest`,
		`metric:   exact_match on "entities_json"`,
		"train:    6 examples",
		"val:      2 examples",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing summary line %q\nfull output:\n%s", want, out)
		}
	}
}
