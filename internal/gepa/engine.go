package gepa

import (
	"context"
	"errors"
	"math/rand"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/program"
)

var (
	ErrOptimizeNotImplemented  = errors.New("gepa optimize loop not implemented")
	ErrEvaluatorNotImplemented = errors.New("rollout evaluator not implemented")
	ErrSelectorNotImplemented  = errors.New("candidate selector not implemented")
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
	Selector  Selector
}

type Evaluator interface {
	Evaluate(ctx context.Context, candidate Candidate, examples []program.Example) ([]ExampleResult, error)
}

type Selector interface {
	SelectCandidate(state State, rng *rand.Rand) (int, error)
}

func Optimize(ctx context.Context, opts Options) (Result, error) {
	_ = ctx
	opts = withDefaults(opts)
	return Result{}, ErrOptimizeNotImplemented
}

func withDefaults(opts Options) Options {
	if opts.Evaluator == nil {
		opts.Evaluator = defaultEvaluator{}
	}
	if opts.Reflector == nil {
		opts.Reflector = defaultReflector{}
	}
	if opts.Selector == nil {
		opts.Selector = defaultSelector{}
	}
	return opts
}

type defaultEvaluator struct{}

func (defaultEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return nil, ErrEvaluatorNotImplemented
}

type defaultSelector struct{}

func (defaultSelector) SelectCandidate(State, *rand.Rand) (int, error) {
	return 0, ErrSelectorNotImplemented
}
