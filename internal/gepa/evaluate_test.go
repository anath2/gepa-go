package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/anath2/gepa-go/internal/program"
)

func TestEvaluateCandidate_AccountsMetricCallsAndMean(t *testing.T) {
	state := newPoolState(testMutationProgram())
	examples := []program.Example{
		{Input: map[string]any{"q": "1"}, Expected: map[string]any{"answer": "a"}},
		{Input: map[string]any{"q": "2"}, Expected: map[string]any{"answer": "b"}},
	}
	eval := fixedScoreEvaluator{score: 0.25}

	summary, err := evaluateCandidate(context.Background(), &state, eval, Candidate{"answer": "seed"}, examples)
	if err != nil {
		t.Fatalf("evaluateCandidate() unexpected error: %v", err)
	}
	if state.MetricCalls != 2 {
		t.Fatalf("MetricCalls = %d, want 2", state.MetricCalls)
	}
	if len(summary.Results) != 2 || len(summary.Scores) != 2 {
		t.Fatalf("summary lengths = results %d scores %d, want 2/2", len(summary.Results), len(summary.Scores))
	}
	if summary.Mean != 0.25 {
		t.Fatalf("Mean = %v, want 0.25", summary.Mean)
	}
	if summary.Scores[0] != 0.25 || summary.Scores[1] != 0.25 {
		t.Fatalf("Scores = %v, want [0.25 0.25]", summary.Scores)
	}
}

func TestEvaluateCandidate_RejectsResultLengthMismatch(t *testing.T) {
	state := newPoolState(testMutationProgram())
	examples := []program.Example{
		{Input: map[string]any{"q": "1"}, Expected: map[string]any{"answer": "a"}},
		{Input: map[string]any{"q": "2"}, Expected: map[string]any{"answer": "b"}},
	}

	_, err := evaluateCandidate(context.Background(), &state, badLengthEvaluator{}, Candidate{"answer": "seed"}, examples)
	if err == nil {
		t.Fatal("evaluateCandidate() error = nil, want result length mismatch")
	}
	if !errors.Is(err, errEvaluatorResultLength) {
		t.Fatalf("evaluateCandidate() error = %v, want errEvaluatorResultLength", err)
	}
	if state.MetricCalls != 0 {
		t.Fatalf("MetricCalls = %d, want 0 when evaluation fails", state.MetricCalls)
	}
}

func TestEvaluateCandidate_PropagatesEvaluatorError(t *testing.T) {
	state := newPoolState(testMutationProgram())
	wantErr := errors.New("evaluator down")
	eval := errEvaluator{err: wantErr}

	_, err := evaluateCandidate(context.Background(), &state, eval, Candidate{"answer": "seed"}, []program.Example{
		{Input: map[string]any{"q": "1"}, Expected: map[string]any{"answer": "a"}},
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("evaluateCandidate() error = %v, want %v", err, wantErr)
	}
}

func TestModuleScoreUsesEvaluationWhenPresent(t *testing.T) {
	result := ExampleResult{
		Score:    1,
		Feedback: "global exact match",
		ModuleTraces: []ModuleTrace{{
			ModuleName: "mod",
			Evaluation: &ModuleEvaluation{Score: 0.4, Feedback: "module feedback", Source: EvalSourceExternalEvaluator},
		}},
	}
	got := moduleScore(result, "mod")
	if got != 0.4 {
		t.Fatalf("moduleScore() = %v, want 0.4", got)
	}
}

func TestModuleScoreFallsBackToGlobalWhenNoEvaluation(t *testing.T) {
	result := ExampleResult{
		Score:    0.25,
		Feedback: "global",
		ModuleTraces: []ModuleTrace{{
			ModuleName: "mod",
		}},
	}
	got := moduleScore(result, "mod")
	if got != 0.25 {
		t.Fatalf("moduleScore() = %v, want 0.25", got)
	}
}

func TestScoresForModuleExtractsPerExampleModuleScores(t *testing.T) {
	results := []ExampleResult{
		{
			Score: 1,
			ModuleTraces: []ModuleTrace{{
				ModuleName: "mod",
				Evaluation: &ModuleEvaluation{Score: 0.2, Source: EvalSourceExternalEvaluator},
			}},
		},
		{
			Score: 0,
			ModuleTraces: []ModuleTrace{{
				ModuleName: "mod",
				Evaluation: &ModuleEvaluation{Score: 0.8, Source: EvalSourceExternalEvaluator},
			}},
		},
	}
	got := scoresForModule(results, "mod")
	want := []float64{0.2, 0.8}
	if len(got) != len(want) {
		t.Fatalf("scoresForModule() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scoresForModule()[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestModuleHasEvaluator(t *testing.T) {
	prog := program.Program{
		Modules: []program.Module{
			{Name: "mod", Evaluator: &program.ModuleEvaluator{Kind: "external", Command: []string{"sh", "-c", "true"}}},
			{Name: "other"},
		},
	}
	if !moduleHasEvaluator(prog, "mod") {
		t.Fatal("moduleHasEvaluator(mod) = false, want true")
	}
	if moduleHasEvaluator(prog, "other") {
		t.Fatal("moduleHasEvaluator(other) = true, want false")
	}
}

type fixedScoreEvaluator struct {
	score float64
}

func (e fixedScoreEvaluator) Evaluate(_ context.Context, _ Candidate, examples []program.Example) ([]ExampleResult, error) {
	out := make([]ExampleResult, len(examples))
	for i := range out {
		out[i] = ExampleResult{Score: e.score, Feedback: "ok"}
	}
	return out, nil
}

type errEvaluator struct {
	err error
}

func (e errEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return nil, e.err
}
