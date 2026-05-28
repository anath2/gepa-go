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
	BatchIndices []int             `json:"batch_indices,omitempty"`
	Examples     []program.Example `json:"examples,omitempty"`
	Results      []ExampleResult   `json:"results,omitempty"`
}

type Reflector interface {
	Propose(ctx context.Context, req ReflectionRequest) (string, error)
}

type defaultReflector struct{}

func (defaultReflector) Propose(context.Context, ReflectionRequest) (string, error) {
	return "", errReflectorNotImplemented
}

func proposeReflection(ctx context.Context, reflector Reflector, req ReflectionRequest) (proposalOutcome, error) {
	instruction, err := reflector.Propose(ctx, req)
	if err != nil {
		return proposalOutcome{Failed: true, Reason: err.Error()}, nil
	}
	if strings.TrimSpace(instruction) == "" {
		return proposalOutcome{Failed: true, Reason: "empty reflected instruction"}, nil
	}
	return proposalOutcome{Instruction: instruction}, nil
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
		fmt.Fprintf(&b, "Score: %.6g\n", req.Results[i].Score)
		fmt.Fprintf(&b, "Feedback: %s\n\n", req.Results[i].Feedback)
	}

	return b.String(), nil
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
