package gepa

import (
	"context"
	"errors"
	"fmt"

	"github.com/anath2/gepa-go/internal/program"
)

var errInvalidOptions = errors.New("gepa options: invalid")

type Options struct {
	Problem

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
	if err := validateOpts(opts); err != nil {
		return Result{}, err
	}

	// Initialize the candidate pool P with the seed program and run-artifact writer (Alg. 1, line 2).
	trainLen := len(opts.Train)
	rng := newRNG(opts.Config.Seed)
	state := newPoolState(opts.Program)
	writer := newRunWriter(opts.RunDir, opts.LogTraces)
	if err := writer.init(); err != nil {
		return Result{}, err
	}

	// Seed candidate scored over the full train set (Alg. 1, lines 3-5).
	// In v0, D_feedback and D_pareto are both train.jsonl, so opts.Train serves both roles.
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

	// Main budget loop (Alg. 1, line 6).
	for iter := 0; ; iter++ {
		if !hasBudget(state.MetricCalls, opts.Config.Budget, batchCost*2) {
			break
		}

		// Select parent from the Pareto frontier (Alg. 1, line 7; Alg. 2).
		parentID, err := selector.selectCandidate(state, rng)
		if err != nil {
			return Result{}, err
		}

		// Select module to mutate (Alg. 1, line 8); round-robin picker in v0.
		module, err := moduleAtIteration(opts.Program, iter)
		if err != nil {
			return Result{}, err
		}
		moduleName := module.Name

		// Sample minibatch M from D_feedback (Alg. 1, line 9).
		batchIndices, err := sampleMinibatch(rng, trainLen, minibatchSize)
		if err != nil {
			return Result{}, err
		}
		batch := examplesAtIndices(opts.Train, batchIndices)

		// Score parent on the minibatch: sigma, the "before" score (Alg. 1, line 13).
		parentPrompts := state.Candidates[parentID].Prompts
		parentEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, parentPrompts, batch)
		if err != nil {
			return Result{}, err
		}

		// Record the proposal-requested event for the run log (not part of Alg. 1).
		if err := writer.proposalRequested(state, parentID, moduleName, batchIndices); err != nil {
			return Result{}, err
		}

		// Reflectively update the module prompt from feedback + traces (Alg. 1, lines 10-12).
		// TODO name and module don't need separate properties
		reflectionReq := ReflectionRequest{
			Candidate:    parentPrompts,
			ParentID:     parentID,
			ModuleName:   moduleName,
			Module:       module,
			BatchIndices: batchIndices,
			Examples:     batch,
			Results:      parentEval.Results,
		}
		proposalOut, err := proposeReflection(ctx, opts.Reflector, reflectionReq)
		if err != nil {
			return Result{}, err
		}
		if proposalOut.Failed {
			if err := writer.proposalFailed(state, parentID, moduleName, batchIndices, proposalOut.Reason); err != nil {
				return Result{}, err
			}
			if err := writer.writeProposalTrace(state, trajectoryRecord{
				ParentID:        parentID,
				ParentIDs:       []int{parentID},
				ProposalKind:    proposalReflection,
				MutatedModule:   moduleName,
				BatchIndices:    append([]int(nil), batchIndices...),
				Accepted:        false,
				Reason:          proposalOut.Reason,
				ParentPrompt:    parentPrompts[moduleName],
				RawResponseText: proposalOut.RawResponseText,
				Examples:        traceExamples(batch, parentEval.Results),
			}); err != nil {
				return Result{}, err
			}
			bumpIteration(&state)
			continue
		}

		// Score the proposal on the same minibatch: sigma', the "after" score (Alg. 1, line 13).
		proposal := mutatedCandidate(parentPrompts, moduleName, proposalOut.Instruction)
		proposalEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, proposal, batch)
		if err != nil {
			return Result{}, err
		}

		if err := writer.proposalEvaluated(state, parentID, moduleName, batchIndices, parentEval.Mean, proposalEval.Mean); err != nil {
			return Result{}, err
		}

		// Acceptance gate: keep the proposal only if sigma' > sigma on the minibatch (Alg. 1, line 14).
		// This is the only acceptance test; the full-train eval below does not gate.
		if !strictlyImproves(parentEval.Mean, proposalEval.Mean) {
			if err := writer.proposalRejected(state, parentID, moduleName, batchIndices, parentEval.Mean, proposalEval.Mean); err != nil {
				return Result{}, err
			}
			if err := writer.writeProposalTrace(state, trajectoryRecord{
				ParentID:             parentID,
				ParentIDs:            []int{parentID},
				ProposalKind:         proposalReflection,
				MutatedModule:        moduleName,
				BatchIndices:         append([]int(nil), batchIndices...),
				Accepted:             false,
				Reason:               rejectReasonNoImprovement,
				ParentMean:           &parentEval.Mean,
				ProposalMean:         &proposalEval.Mean,
				ParentPrompt:         parentPrompts[moduleName],
				ProposedPrompt:       proposal[moduleName],
				RawResponseText:      proposalOut.RawResponseText,
				ExtractedInstruction: proposalOut.Instruction,
				Examples:             traceExamples(batch, proposalEval.Results),
			}); err != nil {
				return Result{}, err
			}
			bumpIteration(&state)
			continue
		}

		if !hasBudget(state.MetricCalls, opts.Config.Budget, trainLen) {
			break
		}

		// Proposal already passed the minibatch gate; this full D_pareto pass is
		// unconditional and only produces per-example scores for the Pareto frontier (Alg. 1, lines 16-18).
		fullEval, err := evaluateCandidate(ctx, &state, opts.Evaluator, proposal, opts.Train)
		if err != nil {
			return Result{}, err
		}

		// Add proposal to the pool with ancestry (Alg. 1, line 15).
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
		if err := writer.writeProposalTrace(state, trajectoryRecord{
			ParentID:             parentID,
			ParentIDs:            []int{parentID},
			ProposalKind:         proposalReflection,
			MutatedModule:        moduleName,
			BatchIndices:         append([]int(nil), batchIndices...),
			Accepted:             true,
			ParentMean:           &parentEval.Mean,
			ProposalMean:         &proposalEval.Mean,
			ParentPrompt:         parentPrompts[moduleName],
			ProposedPrompt:       proposal[moduleName],
			RawResponseText:      proposalOut.RawResponseText,
			ExtractedInstruction: proposalOut.Instruction,
			Examples:             traceExamples(batch, proposalEval.Results),
		}); err != nil {
			return Result{}, err
		}
		bumpIteration(&state)
	}

	// Return the candidate with the best aggregate train/D_pareto score (Alg. 1, line 21).
	result := Result{
		BestCandidate: state.BestCandidate,
		MetricCalls:   state.MetricCalls,
	}
	trainMean, err := meanScore(state.TrainScores[state.BestCandidate])
	if err != nil {
		return Result{}, err
	}
	result.TrainMean = trainMean

	// Final score of the best candidate on the held-out set (paper's test set, not part of
	// Alg. 1); reported only, never used for selection, which relies on the train/D_pareto scores above.
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

	if err := writer.writeFinalState(state); err != nil {
		return Result{}, err
	}
	if err := writer.writeFinalResult(result); err != nil {
		return Result{}, err
	}

	return result, nil
}

func traceExamples(examples []program.Example, results []ExampleResult) []trajectoryExample {
	n := len(examples)
	if len(results) < n {
		n = len(results)
	}
	out := make([]trajectoryExample, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, trajectoryExample{
			Input:        examples[i].Input,
			Expected:     examples[i].Expected,
			Output:       results[i].Output,
			Score:        results[i].Score,
			Feedback:     results[i].Feedback,
			Error:        results[i].Error,
			ModuleTraces: results[i].ModuleTraces,
		})
	}
	return out
}

func withDefaults(opts Options) Options {
	if opts.Reflector == nil {
		opts.Reflector = defaultReflector{}
	}
	return opts
}

func validateOpts(opts Options) error {
	if opts.Evaluator == nil {
		return fmt.Errorf("validate options: evaluator is required: %w", errInvalidOptions)
	}
	if len(opts.Program.Modules) == 0 {
		return fmt.Errorf("validate options: program has no modules: %w", errInvalidOptions)
	}
	if len(opts.Train) == 0 {
		return fmt.Errorf("validate options: train set is empty: %w", errInvalidOptions)
	}
	if opts.Config.Budget <= 0 {
		return fmt.Errorf("validate options: budget must be > 0: %w", errInvalidOptions)
	}
	if opts.Config.MinibatchSize <= 0 {
		return fmt.Errorf("validate options: minibatch_size must be > 0: %w", errInvalidOptions)
	}
	return nil
}
