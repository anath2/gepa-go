package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

type inspectFlags struct {
	format      string
	showTree    bool
	showEvents  bool
}

func newInspectCmd() *cobra.Command {
	f := &inspectFlags{}

	cmd := &cobra.Command{
		Use:           "inspect <run-dir>",
		Short:         "Pretty-print events and tree from a run directory",
		Long:          `Render the optimization event log and the parent DAG / tree of accepted candidates from a runs/<id>/ directory.`,
		Args:          cobra.ExactArgs(1),
		SilenceUsage:  true,
		SilenceErrors: false,
		RunE: func(cmd *cobra.Command, args []string) error {
			dir := args[0]
			info, err := os.Stat(dir)
			if err != nil {
				if errors.Is(err, os.ErrNotExist) {
					return fmt.Errorf("run directory not found at %s", dir)
				}
				return fmt.Errorf("stat %s: %w", dir, err)
			}
			if !info.IsDir() {
				return fmt.Errorf("%s is not a directory", dir)
			}

			switch f.format {
			case "text", "json":
			default:
				return fmt.Errorf("invalid --format %q: expected one of text, json", f.format)
			}

			fmt.Fprintf(cmd.OutOrStdout(), "would inspect %s (format=%s show_tree=%t show_events=%t)\n",
				dir, f.format, f.showTree, f.showEvents)
			return nil
		},
	}

	cmd.Flags().StringVar(&f.format, "format", "text", "Output format: text or json")
	cmd.Flags().BoolVar(&f.showTree, "show-tree", true, "Render the parent DAG / tree")
	cmd.Flags().BoolVar(&f.showEvents, "show-events", true, "Render the events log")

	return cmd
}
