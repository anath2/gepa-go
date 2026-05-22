package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/program"
)

var runE2E = flag.Bool("run-e2e", false, "run high-latency end-to-end tests")

func TestOptimizeChineseNERProblemDefinitionE2E(t *testing.T) {
	if !*runE2E {
		t.Skip("set -run-e2e to run high-latency end-to-end tests")
	}

	fixtureDir := filepath.Join("..", "..", "internal", "testdata", "chinese_ner")
	programPath := filepath.Join(fixtureDir, "program.json")
	configPath := filepath.Join(fixtureDir, "config.json")
	trainPath := filepath.Join(fixtureDir, "train.jsonl")
	valPath := filepath.Join(fixtureDir, "val.jsonl")

	validateChineseNERE2EConfig(t, programPath, configPath)

	out, _, err := runCmd(t, "optimize",
		"--program", programPath,
		"--config", configPath,
		"--train", trainPath,
		"--val", valPath,
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

func validateChineseNERE2EConfig(t *testing.T, programPath, configPath string) {
	t.Helper()

	fmt.Fprintf(os.Stdout, "e2e validation: loading program %s\n", programPath)
	prog, err := program.Load(programPath)
	if err != nil {
		t.Fatalf("program validation failed: %v", err)
	}
	fmt.Fprintf(os.Stdout, "e2e validation: program ok (%d modules, %d tools)\n", len(prog.Modules), len(prog.Tools))

	fmt.Fprintf(os.Stdout, "e2e validation: loading config %s\n", configPath)
	cfg, err := config.Load(configPath)
	if err != nil {
		t.Fatalf("config validation failed: %v", err)
	}
	fmt.Fprintf(os.Stdout, "e2e validation: config ok (budget=%d minibatch=%d seed=%d)\n", cfg.Budget, cfg.MinibatchSize, cfg.Seed)

	fmt.Fprintln(os.Stdout, "e2e validation: checking config metric against program output schema")
	if err := cfg.ValidateAgainstProgram(prog); err != nil {
		t.Fatalf("config/program validation failed: %v", err)
	}

	fmt.Fprintln(os.Stdout, "e2e validation: checking program accumulating-state inputs")
	if err := prog.ValidateAgainstDatasetInputSchema(prog.Modules[0].InputSchema); err != nil {
		t.Fatalf("program state validation failed: %v", err)
	}
	fmt.Fprintln(os.Stdout, "e2e validation: program and config validation complete")
}
