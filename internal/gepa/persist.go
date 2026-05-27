package gepa

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type runArtifacts struct {
	RunDir        string
	StatePath     string
	EventsPath    string
	CandidatesDir string
	ResultPath    string
}

func newRunArtifacts(runDir string) runArtifacts {
	return runArtifacts{
		RunDir:        runDir,
		StatePath:     filepath.Join(runDir, "state.json"),
		EventsPath:    filepath.Join(runDir, "events.jsonl"),
		CandidatesDir: filepath.Join(runDir, "candidates"),
		ResultPath:    filepath.Join(runDir, "result.json"),
	}
}

func ensureRunDir(paths runArtifacts) error {
	if err := os.MkdirAll(paths.RunDir, 0o755); err != nil {
		return fmt.Errorf("ensure run dir: %w", err)
	}
	if err := os.MkdirAll(paths.CandidatesDir, 0o755); err != nil {
		return fmt.Errorf("ensure candidates dir: %w", err)
	}
	return nil
}

func writeState(paths runArtifacts, state poolState) error {
	return atomicWriteJSON(paths.StatePath, state)
}

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

func writeCandidate(paths runArtifacts, id int, record candidateRecord) error {
	file := filepath.Join(paths.CandidatesDir, fmt.Sprintf("%04d.json", id))
	return atomicWriteJSON(file, record)
}

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
