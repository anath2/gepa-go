package gepa

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const (
	eventSeedEvaluated     = "seed_evaluated"
	eventProposalRequested = "proposal_requested"
	eventProposalFailed    = "proposal_failed"
	eventProposalEvaluated = "proposal_evaluated"
	eventCandidateRejected = "candidate_rejected"
	eventCandidateAccepted = "candidate_accepted"
	eventMergeRequested    = "merge_requested"
	eventMergeFailed       = "merge_failed"
	eventMergeEvaluated    = "merge_evaluated"
)

const rejectReasonNoImprovement = "proposal did not strictly improve minibatch mean"

type runArtifacts struct {
	RunDir          string
	StatePath       string
	EventsPath      string
	CandidatesDir   string
	TrajectoriesDir string
	ResultPath      string
}

type runWriter struct {
	paths     runArtifacts
	enabled   bool
	logTraces bool
	// traceByIter tracks next per-iteration trajectory suffix for filenames only.
	traceByIter map[int]int
}

// newRunWriter creates a run artifact writer when a run directory is configured.
func newRunWriter(runDir string, logTraces bool) runWriter {
	if runDir == "" {
		return runWriter{}
	}
	return runWriter{
		paths:       newRunArtifacts(runDir),
		enabled:     true,
		logTraces:   logTraces,
		traceByIter: map[int]int{},
	}
}

// init prepares the run artifact directories before optimization starts.
func (w *runWriter) init() error {
	if !w.enabled {
		return nil
	}
	return ensureRunDir(w.paths, w.logTraces)
}

// persistSeed writes the evaluated seed candidate, state snapshot, and seed event.
func (w *runWriter) persistSeed(state poolState) error {
	if !w.enabled {
		return nil
	}
	if err := writeCandidate(w.paths, 0, state.Candidates[0]); err != nil {
		return err
	}
	if err := writeState(w.paths, state); err != nil {
		return err
	}
	return w.appendRunEvent(eventRecord{
		Type:         eventSeedEvaluated,
		Iteration:    state.Iteration,
		MetricCalls:  state.MetricCalls,
		CandidateID:  0,
		ProposalKind: proposalSeed,
	})
}

// persistAcceptedCandidate writes an accepted proposal and records its acceptance event.
func (w *runWriter) persistAcceptedCandidate(state poolState, id int, parentMean, proposalMean float64, parentID int, moduleName string, batchIndices []int) error {
	if !w.enabled {
		return nil
	}
	if err := writeCandidate(w.paths, id, state.Candidates[id]); err != nil {
		return err
	}
	if err := writeState(w.paths, state); err != nil {
		return err
	}
	accepted := true
	ev := proposalEventContext(state, parentID, moduleName, batchIndices)
	ev.Type = eventCandidateAccepted
	ev.CandidateID = id
	ev.ParentMean = &parentMean
	ev.ProposalMean = &proposalMean
	ev.Accepted = &accepted
	return w.appendRunEvent(ev)
}

func (w *runWriter) proposalRequested(state poolState, parentID int, moduleName string, batchIndices []int) error {
	ev := proposalEventContext(state, parentID, moduleName, batchIndices)
	ev.Type = eventProposalRequested
	return w.appendRunEvent(ev)
}

func (w *runWriter) proposalFailed(state poolState, parentID int, moduleName string, batchIndices []int, reason string) error {
	ev := proposalEventContext(state, parentID, moduleName, batchIndices)
	ev.Type = eventProposalFailed
	ev.Reason = reason
	return w.appendRunEvent(ev)
}

func (w *runWriter) proposalEvaluated(state poolState, parentID int, moduleName string, batchIndices []int, parentMean, proposalMean float64) error {
	ev := proposalEventContext(state, parentID, moduleName, batchIndices)
	ev.Type = eventProposalEvaluated
	ev.ParentMean = &parentMean
	ev.ProposalMean = &proposalMean
	return w.appendRunEvent(ev)
}

func (w *runWriter) mergeRequested(state poolState, parentIDs []int, ancestor int, batchIndices []int) error {
	ev := mergeEventContext(state, parentIDs, batchIndices)
	ev.Type = eventMergeRequested
	ev.Reason = fmt.Sprintf("ancestor=%d", ancestor)
	return w.appendRunEvent(ev)
}

func (w *runWriter) mergeFailed(state poolState, parentIDs []int, batchIndices []int, reason string) error {
	ev := mergeEventContext(state, parentIDs, batchIndices)
	ev.Type = eventMergeFailed
	ev.Reason = reason
	return w.appendRunEvent(ev)
}

func (w *runWriter) mergeEvaluated(state poolState, parentIDs []int, batchIndices []int, parentMean, proposalMean float64) error {
	ev := mergeEventContext(state, parentIDs, batchIndices)
	ev.Type = eventMergeEvaluated
	ev.ParentMean = &parentMean
	ev.ProposalMean = &proposalMean
	return w.appendRunEvent(ev)
}

func (w *runWriter) mergeRejected(state poolState, parentIDs []int, batchIndices []int, parentMean, proposalMean float64) error {
	rejected := false
	ev := mergeEventContext(state, parentIDs, batchIndices)
	ev.Type = eventCandidateRejected
	ev.ParentMean = &parentMean
	ev.ProposalMean = &proposalMean
	ev.Accepted = &rejected
	ev.Reason = rejectReasonNoImprovement
	return w.appendRunEvent(ev)
}

func (w *runWriter) persistAcceptedMerge(state poolState, id int, parentMean, proposalMean float64, parentIDs []int, batchIndices []int) error {
	if !w.enabled {
		return nil
	}
	if err := writeCandidate(w.paths, id, state.Candidates[id]); err != nil {
		return err
	}
	if err := writeState(w.paths, state); err != nil {
		return err
	}
	accepted := true
	ev := mergeEventContext(state, parentIDs, batchIndices)
	ev.Type = eventCandidateAccepted
	ev.CandidateID = id
	ev.ParentMean = &parentMean
	ev.ProposalMean = &proposalMean
	ev.Accepted = &accepted
	return w.appendRunEvent(ev)
}

func mergeEventContext(state poolState, parentIDs []int, batchIndices []int) eventRecord {
	return eventRecord{
		Iteration:    state.Iteration,
		MetricCalls:  state.MetricCalls,
		ParentIDs:    append([]int(nil), parentIDs...),
		ProposalKind: proposalMerge,
		BatchIndices: append([]int(nil), batchIndices...),
	}
}

func (w *runWriter) proposalRejected(state poolState, parentID int, moduleName string, batchIndices []int, parentMean, proposalMean float64) error {
	rejected := false
	ev := proposalEventContext(state, parentID, moduleName, batchIndices)
	ev.Type = eventCandidateRejected
	ev.ParentMean = &parentMean
	ev.ProposalMean = &proposalMean
	ev.Accepted = &rejected
	ev.Reason = rejectReasonNoImprovement
	return w.appendRunEvent(ev)
}

// appendRunEvent appends when RunDir was set; callers use package appendEvent(paths, ...)
// directly for tests and other low-level persistence.
func (w *runWriter) appendRunEvent(event eventRecord) error {
	if !w.enabled {
		return nil
	}
	return appendEvent(w.paths, event)
}

// writeFinalState rewrites state.json with final counters after terminal
// evaluations that did not necessarily accept a new candidate.
func (w *runWriter) writeFinalState(state poolState) error {
	if !w.enabled {
		return nil
	}
	return writeState(w.paths, state)
}

// writeFinalResult writes the completed optimization result artifact.
func (w *runWriter) writeFinalResult(result Result) error {
	if !w.enabled {
		return nil
	}
	return writeResult(w.paths, result)
}

// proposalEventContext builds the common event metadata for reflection proposals.
func proposalEventContext(state poolState, parentID int, moduleName string, batchIndices []int) eventRecord {
	return eventRecord{
		Iteration:     state.Iteration,
		MetricCalls:   state.MetricCalls,
		CandidateID:   parentID,
		ParentIDs:     []int{parentID},
		ProposalKind:  proposalReflection,
		MutatedModule: moduleName,
		BatchIndices:  append([]int(nil), batchIndices...),
	}
}

type trajectoryExample struct {
	Input        map[string]any `json:"input"`
	Expected     map[string]any `json:"expected"`
	Output       map[string]any `json:"output,omitempty"`
	Score        float64        `json:"score"`
	Feedback     string         `json:"feedback"`
	Error        string         `json:"error,omitempty"`
	ModuleTraces []ModuleTrace  `json:"module_traces,omitempty"`
}

type trajectoryRecord struct {
	Iteration            int                 `json:"iteration"`
	AttemptInIteration   int                 `json:"attempt_in_iteration"`
	ParentID             int                 `json:"parent_id"`
	ParentIDs            []int               `json:"parent_ids"`
	ProposalKind         proposalKind        `json:"proposal_kind"`
	MutatedModule        string              `json:"mutated_module"`
	BatchIndices         []int               `json:"batch_indices,omitempty"`
	Accepted             bool                `json:"accepted"`
	Reason               string              `json:"reason,omitempty"`
	ParentMean           *float64            `json:"parent_mean,omitempty"`
	ProposalMean         *float64            `json:"proposal_mean,omitempty"`
	ParentPrompt         string              `json:"parent_prompt,omitempty"`
	ProposedPrompt       string              `json:"proposed_prompt,omitempty"`
	RawResponseText      string              `json:"raw_response_text,omitempty"`
	ExtractedInstruction string              `json:"extracted_instruction,omitempty"`
	ReasoningTrace       string              `json:"reasoning_trace,omitempty"`
	Examples             []trajectoryExample `json:"examples,omitempty"`
}

// newRunArtifacts returns the canonical artifact paths for a run directory.
func newRunArtifacts(runDir string) runArtifacts {
	return runArtifacts{
		RunDir:          runDir,
		StatePath:       filepath.Join(runDir, "state.json"),
		EventsPath:      filepath.Join(runDir, "events.jsonl"),
		CandidatesDir:   filepath.Join(runDir, "candidates"),
		TrajectoriesDir: filepath.Join(runDir, "trajectories"),
		ResultPath:      filepath.Join(runDir, "result.json"),
	}
}

// ensureRunDir creates the directory tree required for run artifacts.
func ensureRunDir(paths runArtifacts, logTraces bool) error {
	if err := os.MkdirAll(paths.RunDir, 0o755); err != nil {
		return fmt.Errorf("ensure run dir: %w", err)
	}
	if err := os.MkdirAll(paths.CandidatesDir, 0o755); err != nil {
		return fmt.Errorf("ensure candidates dir: %w", err)
	}
	if logTraces {
		if err := os.MkdirAll(paths.TrajectoriesDir, 0o755); err != nil {
			return fmt.Errorf("ensure trajectories dir: %w", err)
		}
	}
	return nil
}

// writeState atomically writes the latest pool state snapshot.
func writeState(paths runArtifacts, state poolState) error {
	return atomicWriteJSON(paths.StatePath, state)
}

// appendEvent appends a single JSONL event to the run event log.
func appendEvent(paths runArtifacts, event eventRecord) error {
	if err := os.MkdirAll(filepath.Dir(paths.EventsPath), 0o755); err != nil {
		return fmt.Errorf("append event: %w", err)
	}
	f, err := os.OpenFile(paths.EventsPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("append event: %w", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	if err := enc.Encode(event); err != nil {
		return fmt.Errorf("append event: %w", err)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("append event: %w", err)
	}
	return nil
}

// writeCandidate atomically writes one candidate artifact using a zero-padded ID.
func writeCandidate(paths runArtifacts, id int, record candidateRecord) error {
	file := filepath.Join(paths.CandidatesDir, fmt.Sprintf("%04d.json", id))
	return atomicWriteJSON(file, record)
}

// writeResult atomically writes the final optimization result.
func writeResult(paths runArtifacts, result Result) error {
	return atomicWriteJSON(paths.ResultPath, result)
}

func writeTrajectory(paths runArtifacts, filename string, trace trajectoryRecord) error {
	return atomicWriteJSON(filepath.Join(paths.TrajectoriesDir, filename), trace)
}

func (w *runWriter) writeProposalTrace(state poolState, trace trajectoryRecord) error {
	if !w.enabled || !w.logTraces {
		return nil
	}
	attempt := w.traceByIter[state.Iteration]
	w.traceByIter[state.Iteration] = attempt + 1
	trace.Iteration = state.Iteration
	trace.AttemptInIteration = attempt
	filename := fmt.Sprintf("%04d-%02d.json", state.Iteration, attempt)
	return writeTrajectory(w.paths, filename, trace)
}

func atomicWriteJSON(path string, v any) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("atomic write: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("atomic write: %w", err)
	}
	tmpPath := tmp.Name()

	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		tmp.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("atomic write: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("atomic write: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("atomic write: %w", err)
	}
	return nil
}
