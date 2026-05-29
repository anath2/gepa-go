package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/gepa"
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

	validateChineseNERE2EConfig(t, programPath, configPath, trainPath, valPath)

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

func validateChineseNERE2EConfig(t *testing.T, programPath, configPath, trainPath, valPath string) {
	t.Helper()

	fmt.Fprintf(os.Stdout, "e2e validation: loading problem from %s\n", programPath)
	problem, err := gepa.LoadProblem(gepa.ProblemPaths{
		Program: programPath,
		Config:  configPath,
		Train:   trainPath,
		Val:     valPath,
	})
	if err != nil {
		t.Fatalf("problem validation failed: %v", err)
	}
	fmt.Fprintf(os.Stdout, "e2e validation: problem ok (%d modules, %d tools, train=%d val=%d)\n",
		len(problem.Program.Modules), len(problem.Program.Tools), len(problem.Train), len(problem.Val))
}
