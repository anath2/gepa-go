package gepa

import (
	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/program"
)

// ProblemPaths points at the four user-provided problem-definition files.
type ProblemPaths struct {
	Program string
	Config  string
	Train   string
	Val     string
}

// Problem is the validated optimization problem assembled from program,
// configuration, and train/validation datasets.
type Problem struct {
	Program program.Program
	Config  config.Config
	Train   []program.Example
	Val     []program.Example
}

// LoadProblem loads and cross-validates all four user-provided inputs.
//
// Loader call order intentionally matches cmd/gepa/optimize.go's previous
// Phase-2 sequence so error precedence and integration-test expectations stay
// stable:
// 1) program.Load
// 2) config.Load
// 3) cfg.ValidateAgainstProgram
// 4) prog.ValidateAgainstDatasetInputSchema
// 5) program.LoadDataset(train)
// 6) program.LoadDataset(val)
func LoadProblem(paths ProblemPaths) (Problem, error) {
	prog, err := program.Load(paths.Program)
	if err != nil {
		return Problem{}, err
	}
	cfg, err := config.Load(paths.Config)
	if err != nil {
		return Problem{}, err
	}
	if err := cfg.ValidateAgainstProgram(prog); err != nil {
		return Problem{}, err
	}
	if err := prog.ValidateAgainstDatasetInputSchema(prog.Modules[0].InputSchema); err != nil {
		return Problem{}, err
	}
	train, err := program.LoadDataset(paths.Train, prog.Modules[0].InputSchema, cfg.Metric.Field)
	if err != nil {
		return Problem{}, err
	}
	val, err := program.LoadDataset(paths.Val, prog.Modules[0].InputSchema, cfg.Metric.Field)
	if err != nil {
		return Problem{}, err
	}

	return Problem{
		Program: prog,
		Config:  cfg,
		Train:   train,
		Val:     val,
	}, nil
}
