package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

type inspectFlags struct {
	format     string
	showTree   bool
	showEvents bool
}

type inspectPoolState struct {
	Iteration     int                    `json:"iteration"`
	MetricCalls   int                    `json:"metric_calls"`
	Candidates    []inspectCandidateRec  `json:"candidates"`
	TrainScores   [][]float64            `json:"train_scores"`
	BestCandidate int                    `json:"best_candidate"`
}

type inspectCandidateRec struct {
	ID            int               `json:"id"`
	ParentIDs     []int             `json:"parent_ids"`
	ProposalKind  string            `json:"proposal_kind"`
	MutatedModule string            `json:"mutated_module,omitempty"`
	CreatedAtIter int               `json:"created_at_iter"`
	Prompts       map[string]string `json:"prompts"`
}

type inspectEventRec struct {
	Type          string   `json:"type"`
	Iteration     int      `json:"iteration"`
	MetricCalls   int      `json:"metric_calls"`
	CandidateID   int      `json:"candidate_id,omitempty"`
	ParentIDs     []int    `json:"parent_ids,omitempty"`
	ProposalKind  string   `json:"proposal_kind,omitempty"`
	MutatedModule string   `json:"mutated_module,omitempty"`
	BatchIndices  []int    `json:"batch_indices,omitempty"`
	ParentMean    *float64 `json:"parent_mean,omitempty"`
	ProposalMean  *float64 `json:"proposal_mean,omitempty"`
	Accepted      *bool    `json:"accepted,omitempty"`
	Reason        string   `json:"reason,omitempty"`
}

type inspectResultRec struct {
	BestCandidate     int      `json:"best_candidate"`
	MetricCalls       int      `json:"metric_calls"`
	TrainMean         float64  `json:"train_mean"`
	ValidationMean    *float64 `json:"validation_mean,omitempty"`
	ValidationSkipped string   `json:"validation_skipped,omitempty"`
}

type inspectSummary struct {
	Iteration     int      `json:"iteration"`
	MetricCalls   int      `json:"metric_calls"`
	BestCandidate int      `json:"best_candidate"`
	TrainMean     *float64 `json:"train_mean,omitempty"`
	ValidationMean *float64 `json:"validation_mean,omitempty"`
	ValidationSkipped string `json:"validation_skipped,omitempty"`
}

type inspectCandidate struct {
	ID            int               `json:"id"`
	ParentIDs     []int             `json:"parent_ids"`
	ProposalKind  string            `json:"proposal_kind"`
	MutatedModule string            `json:"mutated_module,omitempty"`
	CreatedAtIter int               `json:"created_at_iter"`
	TrainMean     *float64          `json:"train_mean,omitempty"`
	Prompts       map[string]string `json:"prompts"`
}

type inspectEvent struct {
	Type          string   `json:"type"`
	Iteration     int      `json:"iteration"`
	MetricCalls   int      `json:"metric_calls"`
	CandidateID   int      `json:"candidate_id,omitempty"`
	ParentIDs     []int    `json:"parent_ids,omitempty"`
	ProposalKind  string   `json:"proposal_kind,omitempty"`
	MutatedModule string   `json:"mutated_module,omitempty"`
	BatchIndices  []int    `json:"batch_indices,omitempty"`
	ParentMean    *float64 `json:"parent_mean,omitempty"`
	ProposalMean  *float64 `json:"proposal_mean,omitempty"`
	Accepted      *bool    `json:"accepted,omitempty"`
	Reason        string   `json:"reason,omitempty"`
}

type inspectReport struct {
	RunDir     string             `json:"run_dir"`
	Summary    inspectSummary     `json:"summary"`
	Candidates []inspectCandidate `json:"candidates,omitempty"`
	Events     []inspectEvent     `json:"events,omitempty"`
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

			report, err := loadInspectReport(dir, f.showTree, f.showEvents)
			if err != nil {
				return err
			}

			out := cmd.OutOrStdout()
			switch f.format {
			case "json":
				enc := json.NewEncoder(out)
				enc.SetIndent("", "  ")
				if err := enc.Encode(report); err != nil {
					return fmt.Errorf("encode inspect report: %w", err)
				}
			default:
				renderInspectText(out, report, f.showTree, f.showEvents)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&f.format, "format", "text", "Output format: text or json")
	cmd.Flags().BoolVar(&f.showTree, "show-tree", true, "Render the parent DAG / tree")
	cmd.Flags().BoolVar(&f.showEvents, "show-events", true, "Render the events log")

	return cmd
}

func loadInspectReport(runDir string, showTree, showEvents bool) (inspectReport, error) {
	statePath := filepath.Join(runDir, "state.json")
	stateData, err := os.ReadFile(statePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return inspectReport{}, fmt.Errorf("missing state.json in %s", runDir)
		}
		return inspectReport{}, fmt.Errorf("read state.json: %w", err)
	}

	var state inspectPoolState
	if err := json.Unmarshal(stateData, &state); err != nil {
		return inspectReport{}, fmt.Errorf("decode state.json: %w", err)
	}

	var result *inspectResultRec
	resultPath := filepath.Join(runDir, "result.json")
	if data, err := os.ReadFile(resultPath); err == nil {
		var r inspectResultRec
		if err := json.Unmarshal(data, &r); err != nil {
			return inspectReport{}, fmt.Errorf("decode result.json: %w", err)
		}
		result = &r
	} else if !errors.Is(err, os.ErrNotExist) {
		return inspectReport{}, fmt.Errorf("read result.json: %w", err)
	}

	summary := inspectSummary{
		Iteration:     state.Iteration,
		MetricCalls:   state.MetricCalls,
		BestCandidate: state.BestCandidate,
	}
	if result != nil {
		trainMean := result.TrainMean
		summary.TrainMean = &trainMean
		summary.ValidationMean = result.ValidationMean
		summary.ValidationSkipped = result.ValidationSkipped
	}

	report := inspectReport{
		RunDir:  runDir,
		Summary: summary,
	}

	if showTree {
		report.Candidates = buildInspectCandidates(state)
	}

	if showEvents {
		eventsPath := filepath.Join(runDir, "events.jsonl")
		events, err := readInspectEvents(eventsPath)
		if err != nil {
			return inspectReport{}, err
		}
		report.Events = events
	}

	return report, nil
}

func buildInspectCandidates(state inspectPoolState) []inspectCandidate {
	out := make([]inspectCandidate, len(state.Candidates))
	for i, c := range state.Candidates {
		cand := inspectCandidate{
			ID:            c.ID,
			ParentIDs:     append([]int(nil), c.ParentIDs...),
			ProposalKind:  c.ProposalKind,
			MutatedModule: c.MutatedModule,
			CreatedAtIter: c.CreatedAtIter,
			Prompts:       cloneStringMap(c.Prompts),
		}
		if i < len(state.TrainScores) && len(state.TrainScores[i]) > 0 {
			mean := meanScores(state.TrainScores[i])
			cand.TrainMean = &mean
		}
		out[i] = cand
	}
	return out
}

func meanScores(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	var sum float64
	for _, s := range scores {
		sum += s
	}
	return sum / float64(len(scores))
}

func cloneStringMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func readInspectEvents(path string) ([]inspectEvent, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("read events.jsonl: %w", err)
	}
	defer f.Close()

	var events []inspectEvent
	sc := bufio.NewScanner(f)
	lineNum := 0
	for sc.Scan() {
		lineNum++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var ev inspectEventRec
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return nil, fmt.Errorf("decode events.jsonl line %d: %w", lineNum, err)
		}
		events = append(events, inspectEvent(ev))
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read events.jsonl: %w", err)
	}
	return events, nil
}

func renderInspectText(w io.Writer, report inspectReport, showTree, showEvents bool) {
	s := report.Summary
	fmt.Fprintf(w, "run: %s\n", report.RunDir)
	fmt.Fprintf(w, "iteration=%d metric_calls=%d best_candidate=%d",
		s.Iteration, s.MetricCalls, s.BestCandidate)
	if s.TrainMean != nil {
		fmt.Fprintf(w, " train_mean=%g", *s.TrainMean)
	}
	fmt.Fprintln(w)
	if s.ValidationMean != nil {
		fmt.Fprintf(w, "validation_mean=%g\n", *s.ValidationMean)
	}
	if s.ValidationSkipped != "" {
		fmt.Fprintf(w, "validation_skipped: %s\n", s.ValidationSkipped)
	}

	if showTree && len(report.Candidates) > 0 {
		fmt.Fprintln(w, "candidates:")
		for _, c := range report.Candidates {
			parents := "none"
			if len(c.ParentIDs) > 0 {
				parts := make([]string, len(c.ParentIDs))
				for i, pid := range c.ParentIDs {
					parts[i] = fmt.Sprintf("%04d", pid)
				}
				parents = strings.Join(parts, ",")
			}
			line := fmt.Sprintf("  %04d %s", c.ID, c.ProposalKind)
			if c.MutatedModule != "" {
				line += fmt.Sprintf(" module=%s", c.MutatedModule)
			}
			line += fmt.Sprintf(" parent %s", parents)
			if c.TrainMean != nil {
				line += fmt.Sprintf(" train_mean=%g", *c.TrainMean)
			}
			fmt.Fprintln(w, line)
			for name, prompt := range c.Prompts {
				fmt.Fprintf(w, "    %s: %s\n", name, truncatePrompt(prompt, 120))
			}
		}
	}

	if showEvents && len(report.Events) > 0 {
		fmt.Fprintln(w, "events:")
		for _, ev := range report.Events {
			line := fmt.Sprintf("  iter=%d calls=%d %s", ev.Iteration, ev.MetricCalls, ev.Type)
			if ev.CandidateID != 0 || ev.Type == "seed_evaluated" {
				line += fmt.Sprintf(" candidate=%d", ev.CandidateID)
			}
			if ev.MutatedModule != "" {
				line += fmt.Sprintf(" module=%s", ev.MutatedModule)
			}
			if ev.ParentMean != nil && ev.ProposalMean != nil {
				line += fmt.Sprintf(" parent_mean=%g proposal_mean=%g", *ev.ParentMean, *ev.ProposalMean)
			}
			if ev.Reason != "" {
				line += fmt.Sprintf(" reason=%q", ev.Reason)
			}
			fmt.Fprintln(w, line)
		}
	}
}

func truncatePrompt(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}
