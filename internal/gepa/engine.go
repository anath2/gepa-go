package gepa

import (
	"context"
	"errors"
	"fmt"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/program"
)

var (
	ErrOptimizeNotImplemented  = errors.New("gepa optimize loop not implemented")
	ErrEvaluatorNotImplemented = errors.New("rollout evaluator not implemented")
)

type Options struct {
	Program   program.Program
	Config    config.Config
	Train     []program.Example
	Val       []program.Example
	RunDir    string
	LogTraces bool

	Evaluator Evaluator
	Reflector Reflector
}

type Evaluator interface {
	Evaluate(ctx context.Context, candidate Candidate, examples []program.Example) ([]ExampleResult, error)
}

func Optimize(ctx context.Context, opts Options) (Result, error) {
	opts = withDefaults(opts)

	trainLen := len(opts.Train)
	rng := newRNG(opts.Config.Seed)
	seedCandidate := SeedCandidate(opts.Program)
	state := State{
		Candidates:    []CandidateRecord{NewSeedRecord(opts.Program)},
		TrainScores:   make([][]float64, 1),
		BestCandidate: 0,
	}

	seedResults, err := opts.Evaluator.Evaluate(ctx, seedCandidate, opts.Train)
	if err != nil {
		return Result{}, err
	}
	state.MetricCalls += len(seedResults)
	state.TrainScores[0] = scores(seedResults)
	if err := recomputeBestCandidate(&state); err != nil {
		return Result{}, err
	}

	minibatchSize := opts.Config.MinibatchSize
	batchCost := minibatchCost(trainLen, minibatchSize)
	selector := ParetoSelector{}

	for iter := 0; ; iter++ {
		if !hasBudget(state.MetricCalls, opts.Config.Budget, batchCost*2) {
			break
		}

		parentID, err := selector.SelectCandidate(state, rng)
		if err != nil {
			return Result{}, err
		}

		moduleName, err := moduleNameAtIteration(opts.Program, iter)
		if err != nil {
			return Result{}, err
		}

		batchIndices, err := sampleMinibatch(rng, trainLen, minibatchSize)
		if err != nil {
			return Result{}, err
		}
		batch := examplesAtIndices(opts.Train, batchIndices)

		parentPrompts := state.Candidates[parentID].Prompts
		parentResults, err := opts.Evaluator.Evaluate(ctx, parentPrompts, batch)
		if err != nil {
			return Result{}, err
		}
		state.MetricCalls += len(parentResults)
		parentMean, err := meanScore(scores(parentResults))
		if err != nil {
			return Result{}, err
		}

		newInstruction, err := opts.Reflector.Propose(ctx, ReflectionRequest{
			Candidate:    parentPrompts,
			ParentID:     parentID,
			ModuleName:   moduleName,
			BatchIndices: batchIndices,
			Examples:     batch,
			Results:      parentResults,
		})
		if err != nil {
			state.Iteration++
			continue
		}

		proposal := cloneCandidate(parentPrompts)
		proposal[moduleName] = newInstruction
		proposalResults, err := opts.Evaluator.Evaluate(ctx, proposal, batch)
		if err != nil {
			return Result{}, err
		}
		state.MetricCalls += len(proposalResults)
		proposalMean, err := meanScore(scores(proposalResults))
		if err != nil {
			return Result{}, err
		}

		if !strictlyImproves(parentMean, proposalMean) {
			state.Iteration++
			continue
		}

		if !hasBudget(state.MetricCalls, opts.Config.Budget, trainLen) {
			break
		}

		fullResults, err := opts.Evaluator.Evaluate(ctx, proposal, opts.Train)
		if err != nil {
			return Result{}, err
		}
		state.MetricCalls += len(fullResults)

		newID := len(state.Candidates)
		state.Candidates = append(state.Candidates, CandidateRecord{
			ID:            newID,
			ParentIDs:     []int{parentID},
			ProposalKind:  ProposalReflection,
			MutatedModule: moduleName,
			CreatedAtIter: iter + 1,
			Prompts:       cloneCandidate(proposal),
		})
		state.TrainScores = append(state.TrainScores, scores(fullResults))
		if err := recomputeBestCandidate(&state); err != nil {
			return Result{}, err
		}
		state.Iteration++
	}

	result := Result{
		BestCandidate: state.BestCandidate,
		MetricCalls:   state.MetricCalls,
	}
	trainMean, err := meanScore(state.TrainScores[state.BestCandidate])
	if err != nil {
		return Result{}, err
	}
	result.TrainMean = trainMean

	if len(opts.Val) > 0 && hasBudget(state.MetricCalls, opts.Config.Budget, len(opts.Val)) {
		bestPrompts := state.Candidates[state.BestCandidate].Prompts
		valResults, err := opts.Evaluator.Evaluate(ctx, bestPrompts, opts.Val)
		if err != nil {
			return Result{}, err
		}
		state.MetricCalls += len(valResults)
		valMean, err := meanScore(scores(valResults))
		if err != nil {
			return Result{}, err
		}
		result.ValidationMean = &valMean
		result.MetricCalls = state.MetricCalls
	} else if len(opts.Val) > 0 {
		result.ValidationSkipped = fmt.Sprintf("insufficient budget: need %d metric calls, have %d remaining", len(opts.Val), opts.Config.Budget-state.MetricCalls)
	}

	return result, nil
}

func withDefaults(opts Options) Options {
	if opts.Evaluator == nil {
		opts.Evaluator = defaultEvaluator{}
	}
	if opts.Reflector == nil {
		opts.Reflector = defaultReflector{}
	}
	return opts
}

func recomputeBestCandidate(state *State) error {
	if len(state.Candidates) == 0 {
		return fmt.Errorf("recompute best candidate: empty candidate pool")
	}
	best := 0
	bestMean, err := meanScore(state.TrainScores[0])
	if err != nil {
		return err
	}
	for i := 1; i < len(state.Candidates); i++ {
		mean, err := meanScore(state.TrainScores[i])
		if err != nil {
			return err
		}
		if mean > bestMean {
			best = i
			bestMean = mean
		}
	}
	state.BestCandidate = best
	return nil
}

type defaultEvaluator struct{}

func (defaultEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return nil, ErrEvaluatorNotImplemented
}
