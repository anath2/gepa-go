package program

import (
	"encoding/json"
	"fmt"
	"os"
)

// Program is the declarative description of the compound AI system being
// optimized: an ordered list of modules and a tool registry. Only module
// prompts are mutated by GEPA; everything else (schemas, tool definitions,
// module ordering) is frozen.
type Program struct {
	Modules []Module        `json:"modules"`
	Tools   map[string]Tool `json:"tools,omitempty"`

	// file is the path Program was loaded from. Set by Load; used as the
	// prefix in validation error messages. Validate(...) uses "program.json"
	// when this is empty.
	file string
}

type Module struct {
	Name         string   `json:"name"`
	Prompt       string   `json:"prompt"`
	InputSchema  Schema   `json:"input_schema"`
	OutputSchema Schema   `json:"output_schema"`
	Tools        []string `json:"tools,omitempty"`
	MaxToolSteps int      `json:"max_tool_steps,omitempty"`
}

type Tool struct {
	Name         string   `json:"name"`
	Kind         string   `json:"kind"`
	Description  string   `json:"description"`
	Command      []string `json:"command"`
	InputSchema  Schema   `json:"input_schema"`
	OutputSchema Schema   `json:"output_schema"`
}

// Load reads, strictly parses, and validates a program.json file. Returns a
// fully validated Program; the file path is retained for use in downstream
// validation error messages.
func Load(path string) (Program, error) {
	f, err := os.Open(path)
	if err != nil {
		return Program{}, fmt.Errorf("%s: %w", path, err)
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var p Program
	if err := dec.Decode(&p); err != nil {
		return Program{}, fmt.Errorf("%s: %w", path, err)
	}
	p.file = path
	if err := p.Validate(); err != nil {
		return Program{}, err
	}
	return p, nil
}

func (p Program) label() string {
	if p.file == "" {
		return "program.json"
	}
	return p.file
}

// Validate applies rules P1–P12. The first failing rule produces an error and
// returns; rule order is fixed so messages are deterministic.
func (p Program) Validate() error {
	file := p.label()

	// P1
	if len(p.Modules) == 0 {
		return fmt.Errorf("%s: modules: at least one module required", file)
	}

	seen := make(map[string]bool, len(p.Modules))
	for i, m := range p.Modules {
		// P3
		if m.Name == "" {
			return fmt.Errorf("%s: modules[%d].name: required", file, i)
		}
		// P2
		if seen[m.Name] {
			return fmt.Errorf("%s: modules[%d].name: duplicate module name %q", file, i, m.Name)
		}
		seen[m.Name] = true
		// P8
		if m.MaxToolSteps < 0 {
			return fmt.Errorf("%s: modules[%d].max_tool_steps: must be >= 0", file, i)
		}
		// P9
		if m.InputSchema.Kind != KindObject {
			return fmt.Errorf("%s: modules[%d].input_schema: top-level must be type=object", file, i)
		}
		// P10
		if m.OutputSchema.Kind != KindObject {
			return fmt.Errorf("%s: modules[%d].output_schema: top-level must be type=object", file, i)
		}
		// P5
		for j, tname := range m.Tools {
			if _, ok := p.Tools[tname]; !ok {
				return fmt.Errorf("%s: modules[%d].tools[%d]: unknown tool %q", file, i, j, tname)
			}
		}
	}

	for key, t := range p.Tools {
		// P4
		if t.Name != key {
			return fmt.Errorf("%s: tools[%q].name: map key %q != name field %q", file, key, key, t.Name)
		}
		// P6
		if t.Kind != "external" {
			return fmt.Errorf("%s: tools[%q].kind: only \"external\" supported in v0, got %q", file, key, t.Kind)
		}
		// P7
		if len(t.Command) == 0 {
			return fmt.Errorf("%s: tools[%q].command: required, non-empty", file, key)
		}
		// P11
		if t.InputSchema.Kind != KindObject {
			return fmt.Errorf("%s: tools[%q].input_schema: top-level must be type=object", file, key)
		}
		// P12
		if t.OutputSchema.Kind != KindObject {
			return fmt.Errorf("%s: tools[%q].output_schema: top-level must be type=object", file, key)
		}
	}

	return nil
}

// ValidateAgainstDatasetInputSchema does the accumulating-state static check:
// every input field of module N (N > 0) must be present in firstInput.Fields
// ∪ outputs(0..N-1), with structurally equal type. Later modules' outputs
// overwrite earlier same-named fields, matching runtime semantics.
func (p Program) ValidateAgainstDatasetInputSchema(firstInput Schema) error {
	file := p.label()

	available := make(map[string]Schema, len(firstInput.Fields))
	for k, v := range firstInput.Fields {
		available[k] = v
	}

	for i := 0; i+1 < len(p.Modules); i++ {
		for outName, outSchema := range p.Modules[i].OutputSchema.Fields {
			available[outName] = outSchema
		}
		next := p.Modules[i+1]
		for fieldName, fieldSchema := range next.InputSchema.Fields {
			existing, ok := available[fieldName]
			if !ok {
				return fmt.Errorf("%s: modules[%d].input_schema.%s: not produced by any prior module or dataset input", file, i+1, fieldName)
			}
			if !schemasEqual(fieldSchema, existing) {
				return fmt.Errorf("%s: modules[%d].input_schema.%s: type does not match prior declaration", file, i+1, fieldName)
			}
		}
	}
	return nil
}

func schemasEqual(a, b Schema) bool {
	if a.Kind != b.Kind {
		return false
	}
	switch a.Kind {
	case KindList:
		if a.Item == nil || b.Item == nil {
			return a.Item == b.Item
		}
		return schemasEqual(*a.Item, *b.Item)
	case KindObject:
		if len(a.Fields) != len(b.Fields) {
			return false
		}
		for k, av := range a.Fields {
			bv, ok := b.Fields[k]
			if !ok {
				return false
			}
			if !schemasEqual(av, bv) {
				return false
			}
		}
		return true
	}
	return true
}
