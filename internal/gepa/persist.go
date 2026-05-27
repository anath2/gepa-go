package gepa

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// RunArtifacts holds the canonical paths for engine artifacts under a run directory.
type RunArtifacts struct {
	RunDir        string
	StatePath     string
	EventsPath    string
	CandidatesDir string
	ResultPath    string
}

// NewRunArtifacts derives standard artifact paths from a run directory.
func NewRunArtifacts(runDir string) RunArtifacts {
	return RunArtifacts{
		RunDir:        runDir,
		StatePath:     filepath.Join(runDir, "state.json"),
		EventsPath:    filepath.Join(runDir, "events.jsonl"),
		CandidatesDir: filepath.Join(runDir, "candidates"),
		ResultPath:    filepath.Join(runDir, "result.json"),
	}
}

// EnsureRunDir ensures the run directory and candidates/ subdirectory exist.
func EnsureRunDir(paths RunArtifacts) error {
	if err := os.MkdirAll(paths.RunDir, 0o755); err != nil {
		return fmt.Errorf("ensure run dir: %w", err)
	}
	if err := os.MkdirAll(paths.CandidatesDir, 0o755); err != nil {
		return fmt.Errorf("ensure candidates dir: %w", err)
	}
	return nil
}

// WriteState rewrites state.json atomically.
func WriteState(paths RunArtifacts, state State) error {
	return atomicWriteJSON(paths.StatePath, state)
}

// AppendEvent appends one JSON event per line to events.jsonl.
func AppendEvent(paths RunArtifacts, event Event) error {
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

// WriteCandidate writes a candidate record as candidates/NNNN.json (immutable).
func WriteCandidate(paths RunArtifacts, id int, record CandidateRecord) error {
	file := filepath.Join(paths.CandidatesDir, fmt.Sprintf("%04d.json", id))
	return atomicWriteJSON(file, record)
}

// WriteResult writes result.json at the end of a run.
func WriteResult(paths RunArtifacts, result Result) error {
	return atomicWriteJSON(paths.ResultPath, result)
}

// atomicWriteJSON writes JSON with indentation via write-temp + rename.
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

