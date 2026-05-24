package gepa

import "github.com/anath2/gepa-go/internal/program"

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
