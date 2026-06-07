package gepa

import (
	"context"
	"errors"
	"fmt"

	"github.com/anath2/gepa-go/internal/program"
)

var errEvaluatorResultLength = errors.New("evaluator returned unexpected number of results")

type evaluationSummary struct {
	Results []ExampleResult
	Scores  []float64
	Mean    float64
}

type evaluateScoreMode int

const (
	scoreModeGlobal evaluateScoreMode = iota
	scoreModeModule
)

func evaluateCandidate(ctx context.Context, state *poolState, evaluator Evaluator, candidate Candidate, examples []program.Example) (evaluationSummary, error) {
	return evaluateCandidateWithMode(ctx, state, evaluator, candidate, examples, scoreModeGlobal, "")
}

func evaluateCandidateWithMode(ctx context.Context, state *poolState, evaluator Evaluator, candidate Candidate, examples []program.Example, mode evaluateScoreMode, moduleName string) (evaluationSummary, error) {
	results, err := evaluator.Evaluate(ctx, candidate, examples)
	if err != nil {
		return evaluationSummary{}, err
	}
	if err := ensureResultLength(examples, results); err != nil {
		return evaluationSummary{}, err
	}
	state.MetricCalls += len(results)

	var valueScores []float64
	switch mode {
	case scoreModeModule:
		valueScores = scoresForModule(results, moduleName)
	default:
		valueScores = scores(results)
	}
	mean, err := meanScore(valueScores)
	if err != nil {
		return evaluationSummary{}, err
	}
	return evaluationSummary{
		Results: results,
		Scores:  valueScores,
		Mean:    mean,
	}, nil
}

func ensureResultLength(examples []program.Example, results []ExampleResult) error {
	if len(results) != len(examples) {
		return fmt.Errorf("%w: got %d results for %d examples", errEvaluatorResultLength, len(results), len(examples))
	}
	return nil
}

func moduleScore(result ExampleResult, moduleName string) float64 {
	if trace, ok := selectedModuleTrace(result, moduleName); ok && trace.Evaluation != nil {
		return trace.Evaluation.Score
	}
	return result.Score
}

func scoresForModule(results []ExampleResult, moduleName string) []float64 {
	out := make([]float64, len(results))
	for i, result := range results {
		out[i] = moduleScore(result, moduleName)
	}
	return out
}

func moduleHasEvaluator(prog program.Program, moduleName string) bool {
	for _, module := range prog.Modules {
		if module.Name == moduleName {
			return module.Evaluator != nil
		}
	}
	return false
}

func proposalScoreMode(prog program.Program, moduleName string) (evaluateScoreMode, string) {
	if moduleHasEvaluator(prog, moduleName) {
		return scoreModeModule, moduleName
	}
	return scoreModeGlobal, ""
}

func evaluateProposalCandidate(ctx context.Context, state *poolState, evaluator Evaluator, prog program.Program, candidate Candidate, examples []program.Example, moduleName string) (evaluationSummary, error) {
	mode, scoreModule := proposalScoreMode(prog, moduleName)
	return evaluateCandidateWithMode(ctx, state, evaluator, candidate, examples, mode, scoreModule)
}
