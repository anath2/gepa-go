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

func evaluateCandidate(ctx context.Context, state *poolState, evaluator Evaluator, candidate Candidate, examples []program.Example) (evaluationSummary, error) {
	results, err := evaluator.Evaluate(ctx, candidate, examples)
	if err != nil {
		return evaluationSummary{}, err
	}
	if err := ensureResultLength(examples, results); err != nil {
		return evaluationSummary{}, err
	}
	state.MetricCalls += len(results)
	valueScores := scores(results)
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
