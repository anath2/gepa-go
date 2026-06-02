package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anath2/gepa-go/internal/program"
)

var (
	errReflectorNotImplemented = errors.New("reflection proposer not implemented")
	errReflectionPromptInvalid = errors.New("reflection prompt invalid")
	errInstructionBlockMissing = errors.New("instruction block missing")
)

type ReflectionRequest struct {
	Candidate    Candidate         `json:"candidate"`
	ParentID     int               `json:"parent_id"`
	ModuleName   string            `json:"module_name"`
	Module       program.Module    `json:"module,omitempty"`
	BatchIndices []int             `json:"batch_indices,omitempty"`
	Examples     []program.Example `json:"examples,omitempty"`
	Results      []ExampleResult   `json:"results,omitempty"`
}

type Reflector interface {
	Propose(ctx context.Context, req ReflectionRequest) (string, error)
}

type proposalTrace struct {
	Instruction     string
	RawResponseText string
}

type traceableReflector interface {
	ProposeWithTrace(ctx context.Context, req ReflectionRequest) (proposalTrace, error)
}

type defaultReflector struct{}

func (defaultReflector) Propose(context.Context, ReflectionRequest) (string, error) {
	return "", errReflectorNotImplemented
}

func proposeReflection(ctx context.Context, reflector Reflector, req ReflectionRequest) (proposalOutcome, error) {
	if traced, ok := reflector.(traceableReflector); ok {
		p, err := traced.ProposeWithTrace(ctx, req)
		if err != nil {
			return proposalOutcome{Failed: true, Reason: err.Error()}, nil
		}
		if strings.TrimSpace(p.Instruction) == "" {
			return proposalOutcome{Failed: true, Reason: "empty reflected instruction", RawResponseText: p.RawResponseText}, nil
		}
		return proposalOutcome{Instruction: p.Instruction, RawResponseText: p.RawResponseText}, nil
	}

	instruction, err := reflector.Propose(ctx, req)
	if err != nil {
		return proposalOutcome{Failed: true, Reason: err.Error()}, nil
	}
	if strings.TrimSpace(instruction) == "" {
		return proposalOutcome{Failed: true, Reason: "empty reflected instruction"}, nil
	}
	return proposalOutcome{Instruction: instruction}, nil
}

// ReflectionProposer implements Reflector by rendering the meta-prompt
// from minibatch results and calling an LLM to produce a revised instruction.
type ReflectionProposer struct {
	model ReflectionModel
}

// NewReflectionProposer creates a ReflectionProposer that calls the given model.
func NewReflectionProposer(model ReflectionModel) *ReflectionProposer {
	return &ReflectionProposer{model: model}
}

// Propose renders the reflection meta-prompt, calls the LLM, and extracts
// the new instruction from the first triple-backtick block.
func (rp *ReflectionProposer) Propose(ctx context.Context, req ReflectionRequest) (string, error) {
	trace, err := rp.ProposeWithTrace(ctx, req)
	if err != nil {
		return "", err
	}
	return trace.Instruction, nil
}

func (rp *ReflectionProposer) ProposeWithTrace(ctx context.Context, req ReflectionRequest) (proposalTrace, error) {
	prompt, err := renderReflectionPrompt(req)
	if err != nil {
		return proposalTrace{}, fmt.Errorf("reflection prompt: %w", err)
	}
	text, err := rp.model.Generate(ctx, prompt)
	if err != nil {
		return proposalTrace{}, fmt.Errorf("reflection generate: %w", err)
	}
	instruction, err := extractInstructionBlock(text)
	if err != nil {
		return proposalTrace{}, fmt.Errorf("reflection extract: %w", err)
	}
	return proposalTrace{
		Instruction:     instruction,
		RawResponseText: text,
	}, nil
}

func renderReflectionPrompt(req ReflectionRequest) (string, error) {
	instruction, ok := req.Candidate[req.ModuleName]
	if !ok || strings.TrimSpace(instruction) == "" {
		return "", fmt.Errorf("%w: missing instruction for module %q", errReflectionPromptInvalid, req.ModuleName)
	}
	if len(req.Examples) != len(req.Results) {
		return "", fmt.Errorf("%w: examples/results length mismatch %d != %d", errReflectionPromptInvalid, len(req.Examples), len(req.Results))
	}

	var b strings.Builder
	fmt.Fprintf(&b, "You are improving one module instruction in a compound AI system.\n\n")
	fmt.Fprintf(&b, "Module: %s\n", req.ModuleName)
	fmt.Fprintf(&b, "Current instruction:\n%s\n\n", instruction)
	if req.Module.Name != "" {
		inputSchema, err := prettyJSON(req.Module.InputSchema.ToJSONSchema())
		if err != nil {
			return "", fmt.Errorf("%w: module input schema: %v", errReflectionPromptInvalid, err)
		}
		outputSchema, err := prettyJSON(req.Module.OutputSchema.ToJSONSchema())
		if err != nil {
			return "", fmt.Errorf("%w: module output schema: %v", errReflectionPromptInvalid, err)
		}
		fmt.Fprintf(&b, "Selected module input schema:\n%s\n", inputSchema)
		fmt.Fprintf(&b, "Selected module output schema:\n%s\n", outputSchema)
		fmt.Fprintf(&b, "Do not add fields or responsibilities outside this output schema.\n\n")
	}
	fmt.Fprintf(&b, "Review the minibatch examples, model outputs, scores, and feedback below.\n")
	fmt.Fprintf(&b, "Return only the new instruction inside triple backticks.\n\n")

	for i, example := range req.Examples {
		index := i
		if i < len(req.BatchIndices) {
			index = req.BatchIndices[i]
		}
		input, err := prettyJSON(example.Input)
		if err != nil {
			return "", fmt.Errorf("%w: example %d input: %v", errReflectionPromptInvalid, i, err)
		}
		expected, err := prettyJSON(example.Expected)
		if err != nil {
			return "", fmt.Errorf("%w: example %d expected: %v", errReflectionPromptInvalid, i, err)
		}
		output, err := prettyJSON(req.Results[i].Output)
		if err != nil {
			return "", fmt.Errorf("%w: result %d output: %v", errReflectionPromptInvalid, i, err)
		}

		fmt.Fprintf(&b, "Example %d (dataset index %d)\n", i+1, index)
		fmt.Fprintf(&b, "Input:\n%s\n", input)
		fmt.Fprintf(&b, "Expected:\n%s\n", expected)
		fmt.Fprintf(&b, "Output:\n%s\n", output)
		if trace, ok := selectedModuleTrace(req.Results[i], req.ModuleName); ok {
			traceInput, err := prettyJSON(trace.Input)
			if err != nil {
				return "", fmt.Errorf("%w: result %d module trace input: %v", errReflectionPromptInvalid, i, err)
			}
			traceOutput, err := prettyJSON(trace.Output)
			if err != nil {
				return "", fmt.Errorf("%w: result %d module trace output: %v", errReflectionPromptInvalid, i, err)
			}
			fmt.Fprintf(&b, "Selected module rollout trace:\n")
			fmt.Fprintf(&b, "Selected module input:\n%s\n", traceInput)
			fmt.Fprintf(&b, "Selected module output:\n%s\n", traceOutput)
			if trace.Error != "" {
				fmt.Fprintf(&b, "Selected module error: %s\n", trace.Error)
			}
		}
		fmt.Fprintf(&b, "Score: %.6g\n", req.Results[i].Score)
		fmt.Fprintf(&b, "Feedback: %s\n\n", req.Results[i].Feedback)
	}

	return b.String(), nil
}

func selectedModuleTrace(result ExampleResult, moduleName string) (ModuleTrace, bool) {
	for _, trace := range result.ModuleTraces {
		if trace.ModuleName == moduleName {
			return trace, true
		}
	}
	return ModuleTrace{}, false
}

func extractInstructionBlock(text string) (string, error) {
	start := strings.Index(text, "```")
	if start == -1 {
		return "", errInstructionBlockMissing
	}
	rest := text[start+3:]
	end := strings.Index(rest, "```")
	if end == -1 {
		return "", errInstructionBlockMissing
	}
	block := rest[:end]
	block = strings.TrimPrefix(block, "\r\n")
	block = strings.TrimPrefix(block, "\n")
	if newline := strings.IndexAny(block, "\r\n"); newline > -1 {
		firstLine := strings.TrimSpace(block[:newline])
		if firstLine != "" && !strings.ContainsAny(firstLine, " \t") {
			block = block[newline:]
		}
	}
	return strings.TrimSpace(block), nil
}

func prettyJSON(value any) (string, error) {
	if value == nil {
		return "null", nil
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}
