package config

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/anath2/gepa-go/internal/program"
)

// Config holds GEPA optimizer configuration. Concerns "how to run" — budget,
// models, metric, RNG seed — not "what to optimize" (that's Program).
type Config struct {
	Budget                    int    `json:"budget"`
	MinibatchSize             int    `json:"minibatch_size,omitempty"`
	DefaultMaxToolSteps       int    `json:"default_max_tool_steps,omitempty"`
	Seed                      int64  `json:"seed"`
	ReflectionModel           string `json:"reflection_model"`
	ReflectionReasoningEffort string `json:"reflection_reasoning_effort,omitempty"`
	TaskModel                 string `json:"task_model"`
	TaskReasoningEffort       string `json:"task_reasoning_effort,omitempty"`
	Metric                    Metric `json:"metric"`
	LogDir                    string `json:"log_dir,omitempty"`
	LogTraces                 bool   `json:"log_traces,omitempty"`

	file string
}

type Metric struct {
	Kind  string `json:"kind"`
	Field string `json:"field"`
}

// Load reads, strictly parses, applies defaults to, and validates a config.json
// file. The file path is retained for use in validation error messages.
func Load(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var c Config
	if err := dec.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	c.file = path
	c.applyDefaults()
	if err := c.validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

func (c *Config) applyDefaults() {
	if c.MinibatchSize == 0 {
		c.MinibatchSize = 3
	}
	if c.DefaultMaxToolSteps == 0 {
		c.DefaultMaxToolSteps = 8
	}
	if c.LogDir == "" {
		c.LogDir = "./runs/"
	}
}

func (c Config) label() string {
	if c.file == "" {
		return "config.json"
	}
	return c.file
}

// Validate applies rules C1–C7 in order. Run after applyDefaults so that
// defaulted fields aren't reported as zero. C8 (seed is any int64) is a no-op.
func (c Config) validate() error {
	file := c.label()

	if c.Budget <= 0 {
		return fmt.Errorf("%s: budget: must be > 0", file)
	}
	if c.MinibatchSize <= 0 {
		return fmt.Errorf("%s: minibatch_size: must be > 0", file)
	}
	if c.DefaultMaxToolSteps <= 0 {
		return fmt.Errorf("%s: default_max_tool_steps: must be > 0", file)
	}
	if c.ReflectionModel == "" {
		return fmt.Errorf("%s: reflection_model: required", file)
	}
	if c.TaskModel == "" {
		return fmt.Errorf("%s: task_model: required", file)
	}
	if err := validateReasoningEffort(file, "reflection_reasoning_effort", c.ReflectionReasoningEffort); err != nil {
		return err
	}
	if err := validateReasoningEffort(file, "task_reasoning_effort", c.TaskReasoningEffort); err != nil {
		return err
	}
	if c.Metric.Kind != "exact_match" {
		return fmt.Errorf("%s: metric.kind: only \"exact_match\" supported in v0, got %q", file, c.Metric.Kind)
	}
	if c.Metric.Field == "" {
		return fmt.Errorf("%s: metric.field: required", file)
	}
	return nil
}

func validateReasoningEffort(file, field, value string) error {
	if value == "" {
		return nil
	}
	switch value {
	case "xhigh", "high", "medium", "low", "minimal", "none":
		return nil
	default:
		return fmt.Errorf("%s: %s: must be one of \"xhigh\", \"high\", \"medium\", \"low\", \"minimal\", \"none\", got %q", file, field, value)
	}
}

// ValidateAgainstProgram cross-checks that the metric reads a string field
// that's actually declared in the last module's output_schema.
func (c Config) ValidateAgainstProgram(p program.Program) error {
	file := c.label()
	if len(p.Modules) == 0 {
		return fmt.Errorf("%s: cannot cross-check metric.field: program has no modules", file)
	}
	last := p.Modules[len(p.Modules)-1]
	field, ok := last.OutputSchema.Fields[c.Metric.Field]
	if !ok {
		return fmt.Errorf("%s: metric.field %q not declared in %s: modules[last].output_schema", file, c.Metric.Field, "program.json")
	}
	if field.Kind != program.KindString {
		return fmt.Errorf("%s: metric.field %q must be type string in %s: modules[last].output_schema.%s, got %v", file, c.Metric.Field, "program.json", c.Metric.Field, field.Kind)
	}
	return nil
}
