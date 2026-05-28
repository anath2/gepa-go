package gepa

import (
	"context"
	"errors"
	"fmt"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/program"
)

var errEvaluatorNotImplemented = errors.New("rollout evaluator not implemented")

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
	state := newPoolState(opts.Program)
	writer := newRunWriter(opts.RunDir)
	if err := writer.init(); err != nil {
		return Result{}, err
	}

	seedEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, state.Candidates[0].Prompts, opts.Train)
	if err != nil {
		return Result{}, err
	}
	if err := setSeedTrainScores(&state, trainLen, seedEval.Scores); err != nil {
		return Result{}, err
	}
	if err := writer.persistSeed(state); err != nil {
		return Result{}, err
	}

	minibatchSize := opts.Config.MinibatchSize
	batchCost := minibatchCost(trainLen, minibatchSize)
	selector := paretoSelector{}

	for iter := 0; ; iter++ {
		if !hasBudget(state.MetricCalls, opts.Config.Budget, batchCost*2) {
			break
		}

		parentID, err := selector.selectCandidate(state, rng)
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
		parentEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, parentPrompts, batch)
		if err != nil {
			return Result{}, err
		}

		if err := writer.proposalRequested(state, parentID, moduleName, batchIndices); err != nil {
			return Result{}, err
		}

		proposalOut, err := proposeReflection(ctx, opts.Reflector, ReflectionRequest{
			Candidate:    parentPrompts,
			ParentID:     parentID,
			ModuleName:   moduleName,
			BatchIndices: batchIndices,
			Examples:     batch,
			Results:      parentEval.Results,
		})
		if err != nil {
			return Result{}, err
		}
		if proposalOut.Failed {
			if err := writer.proposalFailed(state, parentID, moduleName, batchIndices, proposalOut.Reason); err != nil {
				return Result{}, err
			}
			bumpIteration(&state)
			continue
		}

		proposal := mutatedCandidate(parentPrompts, moduleName, proposalOut.Instruction)
		proposalEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, proposal, batch)
		if err != nil {
			return Result{}, err
		}

		if err := writer.proposalEvaluated(state, parentID, moduleName, batchIndices, parentEval.Mean, proposalEval.Mean); err != nil {
			return Result{}, err
		}

		if !strictlyImproves(parentEval.Mean, proposalEval.Mean) {
			if err := writer.proposalRejected(state, parentID, moduleName, batchIndices, parentEval.Mean, proposalEval.Mean); err != nil {
				return Result{}, err
			}
			bumpIteration(&state)
			continue
		}

		if !hasBudget(state.MetricCalls, opts.Config.Budget, trainLen) {
			break
		}

		fullEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, proposal, opts.Train)
		if err != nil {
			return Result{}, err
		}
		newID, err := acceptCandidate(&state, trainLen, acceptCandidateParams{
			ParentIDs:     []int{parentID},
			ProposalKind:  proposalReflection,
			MutatedModule: moduleName,
			CreatedAtIter: iter + 1,
			Prompts:       proposal,
			TrainScores:   fullEval.Scores,
		})
		if err != nil {
			return Result{}, err
		}
		if err := writer.persistAcceptedCandidate(state, newID, parentEval.Mean, proposalEval.Mean, parentID, moduleName, batchIndices); err != nil {
			return Result{}, err
		}
		bumpIteration(&state)
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
		valEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, bestPrompts, opts.Val)
		if err != nil {
			return Result{}, err
		}
		result.ValidationMean = &valEval.Mean
		result.MetricCalls = state.MetricCalls
	} else if len(opts.Val) > 0 {
		result.ValidationSkipped = fmt.Sprintf("insufficient budget: need %d metric calls, have %d remaining", len(opts.Val), opts.Config.Budget-state.MetricCalls)
	}

	if err := writer.writeFinalResult(result); err != nil {
		return Result{}, err
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

type defaultEvaluator struct{}

func (defaultEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return nil, errEvaluatorNotImplemented
}
