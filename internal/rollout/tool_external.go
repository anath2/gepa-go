package rollout

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"

	"github.com/anath2/gepa-go/internal/program"
)

func runExternalTool(ctx context.Context, tool program.Tool, input map[string]any) (map[string]any, error) {
	if len(tool.Command) == 0 {
		return nil, errorsf("tool %q command empty", tool.Name)
	}
	payload, err := json.Marshal(input)
	if err != nil {
		return nil, errorsf("marshal tool input: %v", err)
	}

	cmd := exec.CommandContext(ctx, tool.Command[0], tool.Command[1:]...)
	cmd.Stdin = bytes.NewReader(payload)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, errorsf("tool %q failed: %v: %s", tool.Name, err, stderr.String())
	}

	var out map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		return nil, errorsf("tool %q stdout invalid json: %v", tool.Name, err)
	}
	if err := tool.OutputSchema.Validate(out, "output"); err != nil {
		return nil, errorsf("tool %q output invalid: %v", tool.Name, err)
	}
	return out, nil
}

func errorsf(format string, args ...any) error {
	return fmt.Errorf("rollout: "+format, args...)
}
