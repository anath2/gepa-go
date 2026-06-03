package rollout

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/program"
)

var errMalformedCandidate = errors.New("rollout: malformed candidate")

type Evaluator struct {
	Program program.Program
	Config  config.Config
	Model   TaskModel
}

type TaskModel interface {
	Generate(ctx context.Context, req ModuleRequest) (ModuleResponse, error)
}

type ModuleRequest struct {
	ModuleName   string
	Instruction  string
	Input        map[string]any
	OutputSchema map[string]any
}

type ModuleResponse struct {
	Output map[string]any
}

func (e Evaluator) Evaluate(ctx context.Context, candidate gepa.Candidate, examples []program.Example) ([]gepa.ExampleResult, error) {
	if e.Model == nil {
		return nil, errors.New("rollout: nil task model")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([]gepa.ExampleResult, len(examples))
	for i, example := range examples {
		result, err := e.evaluateExample(ctx, candidate, example)
		if err != nil {
			return nil, err
		}
		out[i] = result
	}
	return out, nil
}

// TODO: Add a per example timeout and figure out how to handle when the upstream provider times out
func (e Evaluator) evaluateExample(ctx context.Context, candidate gepa.Candidate, example program.Example) (gepa.ExampleResult, error) {
	state := cloneMap(example.Input)
	var finalOutput map[string]any
	traces := make([]gepa.ModuleTrace, 0, len(e.Program.Modules))

	for _, module := range e.Program.Modules {
		instruction, ok := candidate[module.Name]
		if !ok || strings.TrimSpace(instruction) == "" {
			return gepa.ExampleResult{}, fmt.Errorf("%w: module %q prompt missing or blank", errMalformedCandidate, module.Name)
		}
		moduleInput, err := projectInput(state, module.InputSchema)
		if err != nil {
			return gepa.ExampleResult{}, err
		}
		trace := gepa.ModuleTrace{
			ModuleName: module.Name,
			Input:      cloneMap(moduleInput),
		}
		resp, err := e.Model.Generate(ctx, ModuleRequest{
			ModuleName:   module.Name,
			Instruction:  instruction,
			Input:        moduleInput,
			OutputSchema: module.OutputSchema.ToJSONSchema(),
		})
		if err != nil {
			if errors.Is(err, errDecodeModuleOutput) {
				trace.Error = err.Error()
				traces = append(traces, trace)
				return gepa.ExampleResult{
					Score:        0,
					Feedback:     fmt.Sprintf("module %s output decode failed: %v", module.Name, err),
					Error:        err.Error(),
					ModuleTraces: traces,
				}, nil
			}
			return gepa.ExampleResult{}, err
		}
		if resp.Output == nil {
			resp.Output = map[string]any{}
		}
		trace.Output = cloneMap(resp.Output)
		if err := module.OutputSchema.Validate(resp.Output, "output"); err != nil {
			trace.Error = err.Error()
			traces = append(traces, trace)
			return gepa.ExampleResult{
				Score:        0,
				Feedback:     fmt.Sprintf("module %s output invalid: %v", module.Name, err),
				Output:       resp.Output,
				Error:        err.Error(),
				ModuleTraces: traces,
			}, nil
		}
		traces = append(traces, trace)
		for k, v := range resp.Output {
			state[k] = v
		}
		finalOutput = resp.Output
	}

	result := scoreExactMatch(finalOutput, example.Expected, e.Config.Metric.Field)
	result.ModuleTraces = traces
	return result, nil
}

func projectInput(state map[string]any, schema program.Schema) (map[string]any, error) {
	out := make(map[string]any, len(schema.Fields))
	for name := range schema.Fields {
		v, ok := state[name]
		if !ok {
			return nil, fmt.Errorf("rollout: module input field %q missing from state", name)
		}
		out[name] = v
	}
	return out, nil
}

func cloneMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
