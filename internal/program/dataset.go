package program

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// Example is one row of a dataset. Input is validated against the first
// module's input_schema at load time; Expected is consulted by the metric.
type Example struct {
	Input    map[string]any
	Expected map[string]any
}

// LoadDataset reads a JSONL file, applying rules D1–D6 per row. firstInput is
// the schema each row's `input` is validated against (in v0, this is the first
// module's input_schema). metricField is the key the metric will read from
// each row's `expected`.
func LoadDataset(path string, firstInput Schema, metricField string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	// Default scanner buffer caps at 64KiB; bump to 1MiB so reasonable JSONL
	// rows with long context fields fit. Anything bigger is the user's problem.
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)

	var out []Example
	lineno := 0
	for sc.Scan() {
		lineno++
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}

		var raw map[string]any
		if err := json.Unmarshal(line, &raw); err != nil {
			return nil, fmt.Errorf("%s:%d: invalid JSON: %v", path, lineno, err)
		}

		// D2
		for k := range raw {
			if k != "input" && k != "expected" {
				return nil, fmt.Errorf("%s:%d: unknown key %q (allowed: input, expected)", path, lineno, k)
			}
		}

		// D3
		inputRaw, ok := raw["input"]
		if !ok {
			return nil, fmt.Errorf("%s:%d: input: required object", path, lineno)
		}
		input, ok := inputRaw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%s:%d: input: required object", path, lineno)
		}

		// D4
		expectedRaw, ok := raw["expected"]
		if !ok {
			return nil, fmt.Errorf("%s:%d: expected: required object", path, lineno)
		}
		expected, ok := expectedRaw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%s:%d: expected: required object", path, lineno)
		}

		// D5
		if err := firstInput.Validate(input, "input"); err != nil {
			return nil, fmt.Errorf("%s:%d: %v", path, lineno, err)
		}

		// D6
		if _, ok := expected[metricField]; !ok {
			return nil, fmt.Errorf("%s:%d: expected.%s: required", path, lineno, metricField)
		}

		out = append(out, Example{Input: input, Expected: expected})
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return out, nil
}
