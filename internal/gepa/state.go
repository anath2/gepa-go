package gepa

import (
	"errors"
	"fmt"

	"github.com/anath2/gepa-go/internal/program"
)

var (
	ErrEmptyCandidatePool = errors.New("gepa state: empty candidate pool")
	ErrStateInvariant     = errors.New("gepa state: invariant violation")
)

// Candidate is the prompt set being optimized: module name -> instruction.
type Candidate map[string]string

type ProposalKind string

const (
	ProposalSeed       ProposalKind = "seed"
	ProposalReflection ProposalKind = "reflection"
	ProposalMerge      ProposalKind = "merge"
)

type CandidateRecord struct {
	ID            int          `json:"id"`
	ParentIDs     []int        `json:"parent_ids"`
	ProposalKind  ProposalKind `json:"proposal_kind"`
	MutatedModule string       `json:"mutated_module,omitempty"`
	CreatedAtIter int          `json:"created_at_iter"`
	Prompts       Candidate    `json:"prompts"`
}

type State struct {
	Iteration     int               `json:"iteration"`
	MetricCalls   int               `json:"metric_calls"`
	Candidates    []CandidateRecord `json:"candidates"`
	TrainScores   [][]float64       `json:"train_scores"`
	BestCandidate int               `json:"best_candidate"`
}

type ExampleResult struct {
	Score    float64        `json:"score"`
	Feedback string         `json:"feedback"`
	Output   map[string]any `json:"output,omitempty"`
	Error    string         `json:"error,omitempty"`
}

type Event struct {
	Type          string       `json:"type"`
	Iteration     int          `json:"iteration"`
	MetricCalls   int          `json:"metric_calls"`
	CandidateID   int          `json:"candidate_id,omitempty"`
	ParentIDs     []int        `json:"parent_ids,omitempty"`
	ProposalKind  ProposalKind `json:"proposal_kind,omitempty"`
	MutatedModule string       `json:"mutated_module,omitempty"`
	BatchIndices  []int        `json:"batch_indices,omitempty"`
	ParentMean    *float64     `json:"parent_mean,omitempty"`
	ProposalMean  *float64     `json:"proposal_mean,omitempty"`
	Accepted      *bool        `json:"accepted,omitempty"`
	Reason        string       `json:"reason,omitempty"`
}

type Result struct {
	BestCandidate     int      `json:"best_candidate"`
	MetricCalls       int      `json:"metric_calls"`
	TrainMean         float64  `json:"train_mean"`
	ValidationMean    *float64 `json:"validation_mean,omitempty"`
	ValidationSkipped string   `json:"validation_skipped,omitempty"`
}

// AcceptCandidateParams describes a candidate entering the pool after acceptance.
type AcceptCandidateParams struct {
	ParentIDs     []int
	ProposalKind  ProposalKind
	MutatedModule string
	CreatedAtIter int
	Prompts       Candidate
	TrainScores   []float64
}

func SeedCandidate(prog program.Program) Candidate {
	candidate := make(Candidate, len(prog.Modules))
	for _, module := range prog.Modules {
		candidate[module.Name] = module.Prompt
	}
	return candidate
}

func NewSeedRecord(prog program.Program) CandidateRecord {
	return CandidateRecord{
		ID:            0,
		ParentIDs:     []int{},
		ProposalKind:  ProposalSeed,
		CreatedAtIter: 0,
		Prompts:       SeedCandidate(prog),
	}
}

// NewState returns a pool containing only the unevaluated seed (TrainScores[0] unset).
func NewState(prog program.Program) State {
	return State{
		Candidates:    []CandidateRecord{NewSeedRecord(prog)},
		TrainScores:   make([][]float64, 1),
		BestCandidate: 0,
	}
}

// AddMetricCalls increments the metric-call budget counter.
func AddMetricCalls(s *State, n int) error {
	if n < 0 {
		return fmt.Errorf("add metric calls: n must be >= 0: %w", ErrStateInvariant)
	}
	s.MetricCalls += n
	return nil
}

// BumpIteration advances the outer-loop iteration counter.
func BumpIteration(s *State) {
	s.Iteration++
}

// SetSeedTrainScores assigns full-train scores for candidate 0 and recomputes best.
func SetSeedTrainScores(s *State, trainLen int, trainScores []float64) error {
	if len(s.Candidates) != 1 {
		return fmt.Errorf("set seed train scores: want 1 candidate, got %d: %w", len(s.Candidates), ErrStateInvariant)
	}
	if s.Candidates[0].ID != 0 || s.Candidates[0].ProposalKind != ProposalSeed {
		return fmt.Errorf("set seed train scores: candidate 0 is not seed: %w", ErrStateInvariant)
	}
	if len(trainScores) != trainLen {
		return fmt.Errorf("set seed train scores: got %d scores, want %d: %w", len(trainScores), trainLen, ErrStateInvariant)
	}
	s.TrainScores[0] = append([]float64(nil), trainScores...)
	if err := RecomputeBestCandidate(s); err != nil {
		return err
	}
	return validateState(s, trainLen)
}

// AcceptCandidate appends a candidate and score row, cloning prompts. Returns the new ID.
func AcceptCandidate(s *State, trainLen int, p AcceptCandidateParams) (int, error) {
	if err := validateAcceptParams(s, p); err != nil {
		return 0, err
	}
	if len(p.TrainScores) != trainLen {
		return 0, fmt.Errorf("accept candidate: got %d scores, want %d: %w", len(p.TrainScores), trainLen, ErrStateInvariant)
	}

	newID := len(s.Candidates)
	record := CandidateRecord{
		ID:            newID,
		ParentIDs:     append([]int(nil), p.ParentIDs...),
		ProposalKind:  p.ProposalKind,
		MutatedModule: p.MutatedModule,
		CreatedAtIter: p.CreatedAtIter,
		Prompts:       cloneCandidate(p.Prompts),
	}
	s.Candidates = append(s.Candidates, record)
	s.TrainScores = append(s.TrainScores, append([]float64(nil), p.TrainScores...))

	if err := RecomputeBestCandidate(s); err != nil {
		return 0, err
	}
	if err := validateState(s, trainLen); err != nil {
		return 0, err
	}
	return newID, nil
}

// RecomputeBestCandidate sets BestCandidate to the ID with highest train mean (lowest ID on tie).
func RecomputeBestCandidate(s *State) error {
	if len(s.Candidates) == 0 {
		return fmt.Errorf("recompute best candidate: %w", ErrEmptyCandidatePool)
	}
	best := 0
	bestMean, err := meanScore(s.TrainScores[0])
	if err != nil {
		return err
	}
	for i := 1; i < len(s.Candidates); i++ {
		mean, err := meanScore(s.TrainScores[i])
		if err != nil {
			return err
		}
		if mean > bestMean {
			best = i
			bestMean = mean
		}
	}
	s.BestCandidate = best
	return nil
}

func validateAcceptParams(s *State, p AcceptCandidateParams) error {
	if len(s.Candidates) != len(s.TrainScores) {
		return fmt.Errorf("accept candidate: %d candidates, %d score rows: %w", len(s.Candidates), len(s.TrainScores), ErrStateInvariant)
	}
	newID := len(s.Candidates)
	switch p.ProposalKind {
	case ProposalReflection:
		if len(p.ParentIDs) != 1 {
			return fmt.Errorf("accept candidate: reflection wants 1 parent, got %d: %w", len(p.ParentIDs), ErrStateInvariant)
		}
		if p.MutatedModule == "" {
			return fmt.Errorf("accept candidate: reflection requires mutated_module: %w", ErrStateInvariant)
		}
	case ProposalMerge:
		if len(p.ParentIDs) < 2 {
			return fmt.Errorf("accept candidate: merge wants >= 2 parents, got %d: %w", len(p.ParentIDs), ErrStateInvariant)
		}
	default:
		return fmt.Errorf("accept candidate: unsupported proposal kind %q: %w", p.ProposalKind, ErrStateInvariant)
	}
	for _, parentID := range p.ParentIDs {
		if parentID < 0 || parentID >= newID {
			return fmt.Errorf("accept candidate: parent_id %d out of range: %w", parentID, ErrStateInvariant)
		}
	}
	return nil
}

func validateState(s *State, trainLen int) error {
	if len(s.Candidates) != len(s.TrainScores) {
		return fmt.Errorf("validate state: %d candidates, %d score rows: %w", len(s.Candidates), len(s.TrainScores), ErrStateInvariant)
	}
	for i, c := range s.Candidates {
		if c.ID != i {
			return fmt.Errorf("validate state: candidates[%d].id = %d: %w", i, c.ID, ErrStateInvariant)
		}
		row := s.TrainScores[i]
		if row == nil {
			if i == 0 && len(s.Candidates) == 1 {
				continue
			}
			return fmt.Errorf("validate state: train_scores[%d] unset: %w", i, ErrStateInvariant)
		}
		if len(row) != trainLen {
			return fmt.Errorf("validate state: train_scores[%d] length %d, want %d: %w", i, len(row), trainLen, ErrStateInvariant)
		}
	}
	if len(s.Candidates) > 0 && (s.BestCandidate < 0 || s.BestCandidate >= len(s.Candidates)) {
		return fmt.Errorf("validate state: best_candidate %d out of range: %w", s.BestCandidate, ErrStateInvariant)
	}
	return nil
}

// cloneCandidate returns a shallow copy of a candidate prompt map.
func cloneCandidate(candidate Candidate) Candidate {
	out := make(Candidate, len(candidate))
	for name, prompt := range candidate {
		out[name] = prompt
	}
	return out
}
