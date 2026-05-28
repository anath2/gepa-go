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
)

const rejectReasonNoImprovement = "proposal did not strictly improve minibatch mean"

type runArtifacts struct {
	RunDir        string
	StatePath     string
	EventsPath    string
	CandidatesDir string
	ResultPath    string
}

type runWriter struct {
	paths   runArtifacts
	enabled bool
}

// newRunWriter creates a run artifact writer when a run directory is configured.
func newRunWriter(runDir string) runWriter {
	if runDir == "" {
		return runWriter{}
	}
	return runWriter{
		paths:   newRunArtifacts(runDir),
		enabled: true,
	}
}

// init prepares the run artifact directories before optimization starts.
func (w runWriter) init() error {
	if !w.enabled {
		return nil
	}
	return ensureRunDir(w.paths)
}

// persistSeed writes the evaluated seed candidate, state snapshot, and seed event.
func (w runWriter) persistSeed(state poolState) error {
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
func (w runWriter) persistAcceptedCandidate(state poolState, id int, parentMean, proposalMean float64, parentID int, moduleName string, batchIndices []int) error {
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

// appendRunEvent appends when RunDir was set; callers use package appendEvent(paths, ...)
// directly for tests and other low-level persistence.
func (w runWriter) appendRunEvent(event eventRecord) error {
	if !w.enabled {
		return nil
	}
	return appendEvent(w.paths, event)
}

// writeFinalResult writes the completed optimization result artifact.
func (w runWriter) writeFinalResult(result Result) error {
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

// newRunArtifacts returns the canonical artifact paths for a run directory.
func newRunArtifacts(runDir string) runArtifacts {
	return runArtifacts{
		RunDir:        runDir,
		StatePath:     filepath.Join(runDir, "state.json"),
		EventsPath:    filepath.Join(runDir, "events.jsonl"),
		CandidatesDir: filepath.Join(runDir, "candidates"),
		ResultPath:    filepath.Join(runDir, "result.json"),
	}
}

// ensureRunDir creates the directory tree required for run artifacts.
func ensureRunDir(paths runArtifacts) error {
	if err := os.MkdirAll(paths.RunDir, 0o755); err != nil {
		return fmt.Errorf("ensure run dir: %w", err)
	}
	if err := os.MkdirAll(paths.CandidatesDir, 0o755); err != nil {
		return fmt.Errorf("ensure candidates dir: %w", err)
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
