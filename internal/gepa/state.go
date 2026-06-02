package gepa

import (
	"errors"
	"fmt"

	"github.com/anath2/gepa-go/internal/program"
)

var (
	errEmptyCandidatePool = errors.New("gepa state: empty candidate pool")
	errStateInvariant     = errors.New("gepa state: invariant violation")
)

// Candidate is the prompt set being optimized: module name -> instruction.
type Candidate map[string]string

type proposalKind string

const (
	proposalSeed       proposalKind = "seed"
	proposalReflection proposalKind = "reflection"
	proposalMerge      proposalKind = "merge"
)

type candidateRecord struct {
	ID            int          `json:"id"`
	ParentIDs     []int        `json:"parent_ids"`
	ProposalKind  proposalKind `json:"proposal_kind"`
	MutatedModule string       `json:"mutated_module,omitempty"`
	CreatedAtIter int          `json:"created_at_iter"`
	Prompts       Candidate    `json:"prompts"`
}

type poolState struct {
	Iteration     int               `json:"iteration"`
	MetricCalls   int               `json:"metric_calls"`
	Candidates    []candidateRecord `json:"candidates"`
	TrainScores   [][]float64       `json:"train_scores"`
	BestCandidate int               `json:"best_candidate"`
}

type ExampleResult struct {
	Score        float64        `json:"score"`
	Feedback     string         `json:"feedback"`
	Output       map[string]any `json:"output,omitempty"`
	Error        string         `json:"error,omitempty"`
	ModuleTraces []ModuleTrace  `json:"module_traces,omitempty"`
}

type ModuleTrace struct {
	ModuleName string         `json:"module_name"`
	Input      map[string]any `json:"input,omitempty"`
	Output     map[string]any `json:"output,omitempty"`
	Error      string         `json:"error,omitempty"`
}

type eventRecord struct {
	Type          string       `json:"type"`
	Iteration     int          `json:"iteration"`
	MetricCalls   int          `json:"metric_calls"`
	CandidateID   int          `json:"candidate_id,omitempty"`
	ParentIDs     []int        `json:"parent_ids,omitempty"`
	ProposalKind  proposalKind `json:"proposal_kind,omitempty"`
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

// acceptCandidateParams describes a candidate entering the pool after acceptance.
type acceptCandidateParams struct {
	ParentIDs     []int
	ProposalKind  proposalKind
	MutatedModule string
	CreatedAtIter int
	Prompts       Candidate
	TrainScores   []float64
}

func seedCandidate(prog program.Program) Candidate {
	candidate := make(Candidate, len(prog.Modules))
	for _, module := range prog.Modules {
		candidate[module.Name] = module.Prompt
	}
	return candidate
}

func newSeedRecord(prog program.Program) candidateRecord {
	return candidateRecord{
		ID:            0,
		ParentIDs:     []int{},
		ProposalKind:  proposalSeed,
		CreatedAtIter: 0,
		Prompts:       seedCandidate(prog),
	}
}

// newPoolState returns a pool containing only the unevaluated seed (TrainScores[0] unset).
func newPoolState(prog program.Program) poolState {
	return poolState{
		Candidates:    []candidateRecord{newSeedRecord(prog)},
		TrainScores:   make([][]float64, 1),
		BestCandidate: 0,
	}
}

func addMetricCalls(s *poolState, n int) error {
	if n < 0 {
		return fmt.Errorf("add metric calls: n must be >= 0: %w", errStateInvariant)
	}
	s.MetricCalls += n
	return nil
}

func bumpIteration(s *poolState) {
	s.Iteration++
}

func setSeedTrainScores(s *poolState, trainLen int, trainScores []float64) error {
	if len(s.Candidates) != 1 {
		return fmt.Errorf("set seed train scores: want 1 candidate, got %d: %w", len(s.Candidates), errStateInvariant)
	}
	if s.Candidates[0].ID != 0 || s.Candidates[0].ProposalKind != proposalSeed {
		return fmt.Errorf("set seed train scores: candidate 0 is not seed: %w", errStateInvariant)
	}
	if len(trainScores) != trainLen {
		return fmt.Errorf("set seed train scores: got %d scores, want %d: %w", len(trainScores), trainLen, errStateInvariant)
	}
	s.TrainScores[0] = append([]float64(nil), trainScores...)
	if err := recomputeBestCandidate(s); err != nil {
		return err
	}
	return validateState(s, trainLen)
}

func acceptCandidate(s *poolState, trainLen int, p acceptCandidateParams) (int, error) {
	if err := validateAcceptParams(s, p); err != nil {
		return 0, err
	}
	if len(p.TrainScores) != trainLen {
		return 0, fmt.Errorf("accept candidate: got %d scores, want %d: %w", len(p.TrainScores), trainLen, errStateInvariant)
	}

	newID := len(s.Candidates)
	record := candidateRecord{
		ID:            newID,
		ParentIDs:     append([]int(nil), p.ParentIDs...),
		ProposalKind:  p.ProposalKind,
		MutatedModule: p.MutatedModule,
		CreatedAtIter: p.CreatedAtIter,
		Prompts:       cloneCandidate(p.Prompts),
	}
	s.Candidates = append(s.Candidates, record)
	s.TrainScores = append(s.TrainScores, append([]float64(nil), p.TrainScores...))

	if err := recomputeBestCandidate(s); err != nil {
		return 0, err
	}
	if err := validateState(s, trainLen); err != nil {
		return 0, err
	}
	return newID, nil
}

func recomputeBestCandidate(s *poolState) error {
	if len(s.Candidates) == 0 {
		return fmt.Errorf("recompute best candidate: %w", errEmptyCandidatePool)
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

func validateAcceptParams(s *poolState, p acceptCandidateParams) error {
	if len(s.Candidates) != len(s.TrainScores) {
		return fmt.Errorf("accept candidate: %d candidates, %d score rows: %w", len(s.Candidates), len(s.TrainScores), errStateInvariant)
	}
	newID := len(s.Candidates)
	switch p.ProposalKind {
	case proposalReflection:
		if len(p.ParentIDs) != 1 {
			return fmt.Errorf("accept candidate: reflection wants 1 parent, got %d: %w", len(p.ParentIDs), errStateInvariant)
		}
		if p.MutatedModule == "" {
			return fmt.Errorf("accept candidate: reflection requires mutated_module: %w", errStateInvariant)
		}
	case proposalMerge:
		if len(p.ParentIDs) < 2 {
			return fmt.Errorf("accept candidate: merge wants >= 2 parents, got %d: %w", len(p.ParentIDs), errStateInvariant)
		}
	default:
		return fmt.Errorf("accept candidate: unsupported proposal kind %q: %w", p.ProposalKind, errStateInvariant)
	}
	for _, parentID := range p.ParentIDs {
		if parentID < 0 || parentID >= newID {
			return fmt.Errorf("accept candidate: parent_id %d out of range: %w", parentID, errStateInvariant)
		}
	}
	return nil
}

func validateState(s *poolState, trainLen int) error {
	if len(s.Candidates) != len(s.TrainScores) {
		return fmt.Errorf("validate state: %d candidates, %d score rows: %w", len(s.Candidates), len(s.TrainScores), errStateInvariant)
	}
	for i, c := range s.Candidates {
		if c.ID != i {
			return fmt.Errorf("validate state: candidates[%d].id = %d: %w", i, c.ID, errStateInvariant)
		}
		row := s.TrainScores[i]
		if row == nil {
			if i == 0 && len(s.Candidates) == 1 {
				continue
			}
			return fmt.Errorf("validate state: train_scores[%d] unset: %w", i, errStateInvariant)
		}
		if len(row) != trainLen {
			return fmt.Errorf("validate state: train_scores[%d] length %d, want %d: %w", i, len(row), trainLen, errStateInvariant)
		}
	}
	if len(s.Candidates) > 0 && (s.BestCandidate < 0 || s.BestCandidate >= len(s.Candidates)) {
		return fmt.Errorf("validate state: best_candidate %d out of range: %w", s.BestCandidate, errStateInvariant)
	}
	return nil
}

func cloneCandidate(candidate Candidate) Candidate {
	out := make(Candidate, len(candidate))
	for name, prompt := range candidate {
		out[name] = prompt
	}
	return out
}
