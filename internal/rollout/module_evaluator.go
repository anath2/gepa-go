package rollout

import (
	"context"
	"fmt"

	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/program"
)

// contractModuleEvaluation records a score-0 module evaluation for output contract failures.
func contractModuleEvaluation(source, feedback, errText string) gepa.ModuleEvaluation {
	return gepa.ModuleEvaluation{
		Score:    0,
		Feedback: feedback,
		Source:   source,
		Error:    errText,
	}
}

func decodeFailureEvaluation(moduleName string, err error) gepa.ModuleEvaluation {
	return contractModuleEvaluation(
		gepa.EvalSourceDecode,
		fmt.Sprintf("module %s output decode failed: %v", moduleName, err),
		err.Error(),
	)
}

func schemaFailureEvaluation(moduleName string, err error) gepa.ModuleEvaluation {
	return contractModuleEvaluation(
		gepa.EvalSourceSchema,
		fmt.Sprintf("module %s output invalid: %v", moduleName, err),
		err.Error(),
	)
}

// runModuleEvaluator runs the module's optional external evaluator after output contract checks pass.
// Returns nil when the module has no evaluator configured.
func runModuleEvaluator(ctx context.Context, module program.Module, moduleInput, moduleOutput map[string]any, example program.Example) (*gepa.ModuleEvaluation, error) {
	if module.Evaluator == nil {
		return nil, nil
	}
	eval, err := runExternalEvaluator(ctx, *module.Evaluator, module.Name, moduleInput, moduleOutput, example)
	if err != nil {
		return nil, err
	}
	return &eval, nil
}
