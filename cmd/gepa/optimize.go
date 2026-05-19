package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
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
			}

			summary := map[string]any{
				"cmd":        "optimize",
				"program":    f.program,
				"config":     f.config,
				"train":      f.train,
				"val":        f.val,
				"run_id":     f.runID,
				"resume":     f.resume,
				"log_traces": f.logTraces,
			}
			enc := json.NewEncoder(cmd.OutOrStdout())
			return enc.Encode(summary)
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
