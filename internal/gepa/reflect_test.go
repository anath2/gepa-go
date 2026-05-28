package gepa

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anath2/gepa-go/internal/program"
)

func TestProposeReflection_ReturnsInstruction(t *testing.T) {
	ref := &scriptedReflector{proposal: " improved "}
	req := ReflectionRequest{
		Candidate:  Candidate{"answer": "seed"},
		ParentID:   0,
		ModuleName: "answer",
	}

	out, err := proposeReflection(context.Background(), ref, req)
	if err != nil {
		t.Fatalf("proposeReflection() unexpected error: %v", err)
	}
	if out.Failed {
		t.Fatalf("outcome = failed (%q), want success", out.Reason)
	}
	if out.Instruction != " improved " {
		t.Fatalf("Instruction = %q, want trimmed content preserved", out.Instruction)
	}
}

func TestProposeReflection_ReflectorErrorIsSoftFailure(t *testing.T) {
	ref := &scriptedReflector{err: errors.New("reflection failed")}
	req := ReflectionRequest{ModuleName: "answer"}

	out, err := proposeReflection(context.Background(), ref, req)
	if err != nil {
		t.Fatalf("proposeReflection() unexpected error: %v", err)
	}
	if !out.Failed || out.Reason != "reflection failed" {
		t.Fatalf("outcome = %#v, want failed with reflection failed", out)
	}
}

func TestProposeReflection_EmptyInstructionIsSoftFailure(t *testing.T) {
	ref := &scriptedReflector{proposal: "   "}
	req := ReflectionRequest{ModuleName: "answer"}

	out, err := proposeReflection(context.Background(), ref, req)
	if err != nil {
		t.Fatalf("proposeReflection() unexpected error: %v", err)
	}
	if !out.Failed || out.Reason != "empty reflected instruction" {
		t.Fatalf("outcome = %#v, want empty reflected instruction failure", out)
	}
}

func TestRenderReflectionPromptIncludesInstructionExamplesAndFeedback(t *testing.T) {
	prompt, err := renderReflectionPrompt(ReflectionRequest{
		Candidate:    Candidate{"answerer": "Answer the question using the provided context."},
		ParentID:     0,
		ModuleName:   "answerer",
		BatchIndices: []int{2},
		Examples: []program.Example{
			{
				Input: map[string]any{
					"question": "What color is the sky on a clear day?",
					"context":  "On a clear day, the sky usually appears blue.",
				},
				Expected: map[string]any{
					"answer": "blue",
				},
			},
		},
		Results: []ExampleResult{
			{
				Score:    0,
				Feedback: `expected blue, got green`,
				Output: map[string]any{
					"answer": "green",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("renderReflectionPrompt() unexpected error: %v", err)
	}

	for _, want := range []string{
		"Answer the question using the provided context.",
		"What color is the sky on a clear day?",
		"answer",
		"expected blue, got green",
		"Return only the new instruction inside triple backticks",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("rendered prompt missing %q\n--- prompt ---\n%s", want, prompt)
		}
	}
}

func TestRenderReflectionPromptRequiresCurrentModuleInstruction(t *testing.T) {
	_, err := renderReflectionPrompt(ReflectionRequest{
		Candidate:  Candidate{"retriever": "Retrieve supporting context."},
		ModuleName: "answerer",
	})
	if err == nil {
		t.Fatal("renderReflectionPrompt() error = nil, want missing instruction error")
	}
	if !errors.Is(err, errReflectionPromptInvalid) {
		t.Fatalf("renderReflectionPrompt() error = %v, want errReflectionPromptInvalid", err)
	}
}

func TestRenderReflectionPromptRequiresAlignedExamplesAndResults(t *testing.T) {
	_, err := renderReflectionPrompt(ReflectionRequest{
		Candidate:  Candidate{"answerer": "Answer questions."},
		ModuleName: "answerer",
		Examples: []program.Example{
			{Input: map[string]any{"question": "What is 2+2?"}},
		},
	})
	if err == nil {
		t.Fatal("renderReflectionPrompt() error = nil, want alignment error")
	}
	if !errors.Is(err, errReflectionPromptInvalid) {
		t.Fatalf("renderReflectionPrompt() error = %v, want errReflectionPromptInvalid", err)
	}
}

func TestExtractInstructionBlockUsesFirstTripleBacktickBlock(t *testing.T) {
	got, err := extractInstructionBlock("analysis\n```text\nUse context carefully.\n```\n```ignored\nsecond\n```")
	if err != nil {
		t.Fatalf("extractInstructionBlock() unexpected error: %v", err)
	}
	want := "Use context carefully."
	if got != want {
		t.Fatalf("extractInstructionBlock() = %q, want %q", got, want)
	}
}

func TestExtractInstructionBlockRejectsMissingBlock(t *testing.T) {
	_, err := extractInstructionBlock("Use context carefully.")
	if err == nil {
		t.Fatal("extractInstructionBlock() error = nil, want missing block error")
	}
	if !errors.Is(err, errInstructionBlockMissing) {
		t.Fatalf("extractInstructionBlock() error = %v, want errInstructionBlockMissing", err)
	}
}
