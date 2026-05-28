package rollout

import (
	"fmt"

	"github.com/anath2/gepa-go/internal/gepa"
)

func scoreExactMatch(output map[string]any, expected map[string]any, field string) gepa.ExampleResult {
	want, wantOK := expected[field]
	got, gotOK := output[field]
	if !wantOK {
		return gepa.ExampleResult{
			Score:    0,
			Feedback: fmt.Sprintf("expected <missing>, got %s", renderValue(got, gotOK)),
			Error:    fmt.Sprintf("expected.%s missing", field),
			Output:   output,
		}
	}
	if !gotOK {
		return gepa.ExampleResult{
			Score:    0,
			Feedback: fmt.Sprintf("expected %s, got <missing>", renderValue(want, true)),
			Error:    fmt.Sprintf("output.%s missing", field),
			Output:   output,
		}
	}
	if valuesEqual(got, want) {
		return gepa.ExampleResult{
			Score:    1,
			Feedback: "exact match",
			Output:   output,
		}
	}
	return gepa.ExampleResult{
		Score:    0,
		Feedback: fmt.Sprintf("expected %s, got %s", renderValue(want, true), renderValue(got, true)),
		Output:   output,
	}
}

func valuesEqual(got any, want any) bool {
	return fmt.Sprintf("%v", got) == fmt.Sprintf("%v", want)
}

func renderValue(v any, ok bool) string {
	if !ok {
		return "<missing>"
	}
	return fmt.Sprintf("%q", fmt.Sprintf("%v", v))
}
