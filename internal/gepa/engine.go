package gepa

import (
	"context"
	"errors"

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
	return opts
}

type defaultEvaluator struct{}

func (defaultEvaluator) Evaluate(context.Context, Candidate, []program.Example) ([]ExampleResult, error) {
	return nil, ErrEvaluatorNotImplemented
}

