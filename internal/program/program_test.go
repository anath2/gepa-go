package program

import (
	"os"
	"strings"
	"testing"
)

// validProgram returns a minimal valid Program that all P-rule tests start from
// and then mutate to trigger a specific failure.
func validProgram() Program {
	return Program{
		Modules: []Module{
			{
				Name:         "answer",
				Prompt:       "Answer the question.",
				InputSchema:  Schema{Kind: KindObject, Fields: map[string]Schema{"question": {Kind: KindString}}},
				OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"answer": {Kind: KindString}}},
			},
		},
	}
}

func validTool() Tool {
	return Tool{
		Name:         "fetch",
		Kind:         "external",
		Description:  "Fetch something.",
		Command:      []string{"python", "fetch.py"},
		InputSchema:  Schema{Kind: KindObject, Fields: map[string]Schema{"title": {Kind: KindString}}},
		OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"text": {Kind: KindString}}},
	}
}

func TestLoadProgramOK(t *testing.T) {
	p, err := Load("testdata/program_ok.json")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(p.Modules) != 2 {
		t.Errorf("expected 2 modules, got %d", len(p.Modules))
	}
	if _, ok := p.Tools["fetch_doc"]; !ok {
		t.Errorf("expected fetch_doc tool present")
	}
}

func TestValidateP1NoModules(t *testing.T) {
	p := Program{Modules: nil}
	err := p.validate()
	want := "program.json: modules: at least one module required"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP2DuplicateModuleName(t *testing.T) {
	p := validProgram()
	p.Modules = append(p.Modules, p.Modules[0])
	err := p.validate()
	want := `program.json: modules[1].name: duplicate module name "answer"`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP3EmptyName(t *testing.T) {
	p := validProgram()
	p.Modules[0].Name = ""
	err := p.validate()
	want := "program.json: modules[0].name: required"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP4ToolMapKeyMismatch(t *testing.T) {
	p := validProgram()
	t1 := validTool()
	t1.Name = "actually_different"
	p.Tools = map[string]Tool{"fetch": t1}
	err := p.validate()
	want := `program.json: tools["fetch"].name: map key "fetch" != name field "actually_different"`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP5UnknownTool(t *testing.T) {
	p := validProgram()
	p.Modules[0].Tools = []string{"missing"}
	err := p.validate()
	want := `program.json: modules[0].tools[0]: unknown tool "missing"`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP6BadKind(t *testing.T) {
	p := validProgram()
	t1 := validTool()
	t1.Kind = "http"
	p.Tools = map[string]Tool{"fetch": t1}
	err := p.validate()
	want := `program.json: tools["fetch"].kind: only "external" supported in v0, got "http"`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP7EmptyCommand(t *testing.T) {
	p := validProgram()
	t1 := validTool()
	t1.Command = nil
	p.Tools = map[string]Tool{"fetch": t1}
	err := p.validate()
	want := `program.json: tools["fetch"].command: required, non-empty`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP8NegativeMaxToolSteps(t *testing.T) {
	p := validProgram()
	p.Modules[0].MaxToolSteps = -1
	err := p.validate()
	want := "program.json: modules[0].max_tool_steps: must be >= 0"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP9InputSchemaNotObject(t *testing.T) {
	p := validProgram()
	p.Modules[0].InputSchema = Schema{Kind: KindString}
	err := p.validate()
	want := "program.json: modules[0].input_schema: top-level must be type=object"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP10OutputSchemaNotObject(t *testing.T) {
	p := validProgram()
	p.Modules[0].OutputSchema = Schema{Kind: KindString}
	err := p.validate()
	want := "program.json: modules[0].output_schema: top-level must be type=object"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP11ToolInputNotObject(t *testing.T) {
	p := validProgram()
	t1 := validTool()
	t1.InputSchema = Schema{Kind: KindString}
	p.Tools = map[string]Tool{"fetch": t1}
	err := p.validate()
	want := `program.json: tools["fetch"].input_schema: top-level must be type=object`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestValidateP12ToolOutputNotObject(t *testing.T) {
	p := validProgram()
	t1 := validTool()
	t1.OutputSchema = Schema{Kind: KindString}
	p.Tools = map[string]Tool{"fetch": t1}
	err := p.validate()
	want := `program.json: tools["fetch"].output_schema: top-level must be type=object`
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestAccumulatingStatePositive(t *testing.T) {
	// module 0: input {question}; output {docs}
	// module 1: input {question, docs}; output {answer}
	p := Program{
		Modules: []Module{
			{
				Name:         "retriever",
				InputSchema:  Schema{Kind: KindObject, Fields: map[string]Schema{"question": {Kind: KindString}}},
				OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"docs": {Kind: KindList, Item: &Schema{Kind: KindString}}}},
			},
			{
				Name: "answerer",
				InputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{
					"question": {Kind: KindString},
					"docs":     {Kind: KindList, Item: &Schema{Kind: KindString}},
				}},
				OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"answer": {Kind: KindString}}},
			},
		},
	}
	if err := p.ValidateAgainstDatasetInputSchema(p.Modules[0].InputSchema); err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestAccumulatingStateMissingField(t *testing.T) {
	// module 1 expects "summary" but no one produces it.
	p := Program{
		Modules: []Module{
			{
				Name:         "m0",
				InputSchema:  Schema{Kind: KindObject, Fields: map[string]Schema{"question": {Kind: KindString}}},
				OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"docs": {Kind: KindString}}},
			},
			{
				Name:         "m1",
				InputSchema:  Schema{Kind: KindObject, Fields: map[string]Schema{"summary": {Kind: KindString}}},
				OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"answer": {Kind: KindString}}},
			},
		},
	}
	err := p.ValidateAgainstDatasetInputSchema(p.Modules[0].InputSchema)
	want := "program.json: modules[1].input_schema.summary: not produced by any prior module or dataset input"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestAccumulatingStateTypeMismatch(t *testing.T) {
	// module 0 outputs docs:string; module 1 declares docs as list[string].
	p := Program{
		Modules: []Module{
			{
				Name:         "m0",
				InputSchema:  Schema{Kind: KindObject, Fields: map[string]Schema{"question": {Kind: KindString}}},
				OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"docs": {Kind: KindString}}},
			},
			{
				Name: "m1",
				InputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{
					"docs": {Kind: KindList, Item: &Schema{Kind: KindString}},
				}},
				OutputSchema: Schema{Kind: KindObject, Fields: map[string]Schema{"answer": {Kind: KindString}}},
			},
		},
	}
	err := p.ValidateAgainstDatasetInputSchema(p.Modules[0].InputSchema)
	want := "program.json: modules[1].input_schema.docs: type does not match prior declaration"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}
}

func TestLoadProgramUnknownField(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/p.json"
	if err := os.WriteFile(path, []byte(`{"modules":[],"surprise":42}`), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "unknown field") {
		t.Errorf("expected unknown-field error, got: %v", err)
	}
}
