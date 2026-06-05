package rollout

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"os/exec"

	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/program"
)

type externalEvaluatorRequest struct {
	ModuleName string         `json:"module_name"`
	Input      map[string]any `json:"input"`
	Output     map[string]any `json:"output"`
	Example    struct {
		Input    map[string]any `json:"input"`
		Expected map[string]any `json:"expected"`
	} `json:"example"`
}

type externalEvaluatorResponse struct {
	Score    float64 `json:"score"`
	Feedback string  `json:"feedback"`
}

// runExternalEvaluator executes a declared external module evaluator command.
// Nonzero exit is treated as a task-level score-0 result; malformed stdout is a hard error.
func runExternalEvaluator(ctx context.Context, evaluator program.ModuleEvaluator, moduleName string, moduleInput, moduleOutput map[string]any, example program.Example) (gepa.ModuleEvaluation, error) {
	if len(evaluator.Command) == 0 {
		return gepa.ModuleEvaluation{}, errorsf("external evaluator command empty")
	}

	var req externalEvaluatorRequest
	req.ModuleName = moduleName
	req.Input = moduleInput
	req.Output = moduleOutput
	req.Example.Input = example.Input
	req.Example.Expected = example.Expected

	payload, err := json.Marshal(req)
	if err != nil {
		return gepa.ModuleEvaluation{}, errorsf("marshal external evaluator input: %v", err)
	}

	cmd := exec.CommandContext(ctx, evaluator.Command[0], evaluator.Command[1:]...)
	cmd.Stdin = bytes.NewReader(payload)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		feedback := stderr.String()
		if feedback == "" {
			feedback = err.Error()
		}
		return gepa.ModuleEvaluation{
			Score:    0,
			Feedback: feedback,
			Source:   gepa.EvalSourceExternalEvaluator,
			Error:    err.Error(),
		}, nil
	}

	var resp externalEvaluatorResponse
	if err := json.Unmarshal(stdout.Bytes(), &resp); err != nil {
		return gepa.ModuleEvaluation{}, errorsf("external evaluator stdout invalid json: %v", err)
	}
	if math.IsNaN(resp.Score) || math.IsInf(resp.Score, 0) {
		return gepa.ModuleEvaluation{}, errorsf("external evaluator score must be finite, got %v", resp.Score)
	}
	if resp.Score < 0 || resp.Score > 1 {
		return gepa.ModuleEvaluation{}, errorsf("external evaluator score must be in [0,1], got %v", resp.Score)
	}
	if resp.Feedback == "" {
		return gepa.ModuleEvaluation{}, errorsf("external evaluator feedback required")
	}
	return gepa.ModuleEvaluation{
		Score:    resp.Score,
		Feedback: resp.Feedback,
		Source:   gepa.EvalSourceExternalEvaluator,
	}, nil
}
