package main

import (
	"os"

	"github.com/spf13/cobra"
)

const Version = "0.0.0-dev"

func newRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "gepa",
		Short: "GEPA: Reflective prompt optimizer for compound AI systems",
		Long: `GEPA evolves the natural-language prompts of a compound AI system using LLM-based
reflection on execution traces and Pareto-based candidate selection. Based on
Agrawal et al., arXiv:2507.19457. Subcommands: optimize, inspect.`,
		Version:       Version,
		SilenceUsage:  true,
		SilenceErrors: false,
	}
	cmd.AddCommand(newOptimizeCmd())
	cmd.AddCommand(newInspectCmd())
	return cmd
}

func main() {
	if err := newRootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}
