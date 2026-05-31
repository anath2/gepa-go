package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/llm"
	"github.com/anath2/gepa-go/internal/rollout"
)

type optimizeFlags struct {
	program   string
	config    string
	train     string
	val       string
	runID     string
	resume    string
	logTraces bool
}

func newOptimizeCmd() *cobra.Command {
	f := &optimizeFlags{}

	cmd := &cobra.Command{
		Use:           "optimize",
		Short:         "Run the GEPA optimization loop on a program",
		Long:          `Optimize the prompts of a compound AI system declared in program.json against a dataset, using a budgeted reflective evolutionary loop.`,
		SilenceUsage:  true,
		SilenceErrors: false,
		PreRunE: func(cmd *cobra.Command, args []string) error {
			if f.resume == "" {
				var missing []string
				if f.program == "" {
					missing = append(missing, "--program")
				}
				if f.config == "" {
					missing = append(missing, "--config")
				}
				if f.train == "" {
					missing = append(missing, "--train")
				}
				if f.val == "" {
					missing = append(missing, "--val")
				}
				if len(missing) > 0 {
					return fmt.Errorf("required flag(s) %s not set", strings.Join(missing, ", "))
				}
			}
			return nil
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			pathChecks := []struct {
				label string
				path  string
			}{
				{"program.json", f.program},
				{"config.json", f.config},
				{"train.jsonl", f.train},
				{"val.jsonl", f.val},
			}
			for _, pc := range pathChecks {
				if pc.path == "" {
					continue
				}
				info, err := os.Stat(pc.path)
				if err != nil {
					if errors.Is(err, os.ErrNotExist) {
						return fmt.Errorf("%s not found at %s", pc.label, pc.path)
					}
					return fmt.Errorf("stat %s: %w", pc.path, err)
				}
				if info.IsDir() {
					return fmt.Errorf("%s is a directory, expected file: %s", pc.label, pc.path)
				}
			}

			if f.resume != "" {
				info, err := os.Stat(f.resume)
				if err != nil {
					if errors.Is(err, os.ErrNotExist) {
						return fmt.Errorf("run directory not found at %s", f.resume)
					}
					return fmt.Errorf("stat %s: %w", f.resume, err)
				}
				if !info.IsDir() {
					return fmt.Errorf("%s is not a directory", f.resume)
				}
				fmt.Fprintf(cmd.OutOrStdout(), "resume: %s (snapshot reading deferred to Phase 3)\n", f.resume)
				return nil
			}

			problem, err := gepa.LoadProblem(gepa.ProblemPaths{
				Program: f.program,
				Config:  f.config,
				Train:   f.train,
				Val:     f.val,
			})
			if err != nil {
				return err
			}

			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "program:  %d modules, %d tools\n", len(problem.Program.Modules), len(problem.Program.Tools))
			fmt.Fprintf(out, "config:   budget=%d minibatch=%d seed=%d\n", problem.Config.Budget, problem.Config.MinibatchSize, problem.Config.Seed)
			fmt.Fprintf(out, "models:   task=%s  reflection=%s\n", problem.Config.TaskModel, problem.Config.ReflectionModel)
			fmt.Fprintf(out, "metric:   %s on %q\n", problem.Config.Metric.Kind, problem.Config.Metric.Field)
			fmt.Fprintf(out, "train:    %d examples\n", len(problem.Train))
			fmt.Fprintf(out, "val:      %d examples\n", len(problem.Val))

			runID := resolveRunID(f.runID)
			runDir, err := prepareRunDir(problem.Config.LogDir, runID, f.program, f.config)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "run:      %s\n", runDir)

			logTraces := f.logTraces || problem.Config.LogTraces
			result, err := runOptimize(context.Background(), problem, runDir, logTraces)
			if err != nil {
				return err
			}

			fmt.Fprintf(out, "best:     candidate %d  train_mean=%.6g  metric_calls=%d\n",
				result.BestCandidate, result.TrainMean, result.MetricCalls)
			if result.ValidationMean != nil {
				fmt.Fprintf(out, "val_mean: %.6g\n", *result.ValidationMean)
			}
			if result.ValidationSkipped != "" {
				fmt.Fprintf(out, "val:      skipped (%s)\n", result.ValidationSkipped)
			}
			return nil
		},
	}

	cmd.Flags().StringVarP(&f.program, "program", "p", "", "Path to program.json")
	cmd.Flags().StringVarP(&f.config, "config", "c", "", "Path to config.json")
	cmd.Flags().StringVar(&f.train, "train", "", "Path to train.jsonl")
	cmd.Flags().StringVar(&f.val, "val", "", "Path to val.jsonl")
	cmd.Flags().StringVar(&f.runID, "run-id", "", "Override default run directory name (default: timestamp)")
	cmd.Flags().StringVar(&f.resume, "resume", "", "Path to existing run directory to resume from")
	cmd.Flags().BoolVar(&f.logTraces, "log-traces", false, "Emit full LLM trajectories to runs/<id>/trajectories/")

	return cmd
}

// resolveRunID returns flag if non-empty, otherwise a compact timestamp.
func resolveRunID(flag string) string {
	if flag != "" {
		return flag
	}
	return time.Now().Format("20060102-150405")
}

// prepareRunDir creates <logDir>/<runID> and snapshots program.json and
// config.json into it. Returns the run directory path.
func prepareRunDir(logDir, runID, programPath, configPath string) (string, error) {
	runDir := filepath.Join(logDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return "", fmt.Errorf("create run dir: %w", err)
	}
	for _, pair := range []struct {
		src  string
		name string
	}{
		{programPath, "program.json"},
		{configPath, "config.json"},
	} {
		if err := copyFile(pair.src, filepath.Join(runDir, pair.name)); err != nil {
			return "", fmt.Errorf("snapshot %s: %w", pair.name, err)
		}
	}
	return runDir, nil
}

// copyFile copies a file from src to dst, with error-checked close.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Close()
}

// runOptimize wires the LLM client and rollout evaluator and runs the GEPA
// optimization loop.
func runOptimize(ctx context.Context, problem gepa.Problem, runDir string, logTraces bool) (gepa.Result, error) {
	client, err := llm.NewClient()
	if err != nil {
		return gepa.Result{}, err
	}

	taskModel := llm.Model{Name: problem.Config.TaskModel, Client: client}
	reflectionModel := llm.Model{Name: problem.Config.ReflectionModel, Client: client}

	evaluator := rollout.Evaluator{
		Program: problem.Program,
		Config:  problem.Config,
		Model:   rollout.NewLLMTaskModel(taskModel),
	}

	reflector := gepa.NewReflectionProposer(gepa.NewLLMReflectionModel(reflectionModel))

	return gepa.Optimize(ctx, gepa.Options{
		Problem:   problem,
		RunDir:    runDir,
		LogTraces: logTraces,
		Evaluator: evaluator,
		Reflector: reflector,
	})
}
