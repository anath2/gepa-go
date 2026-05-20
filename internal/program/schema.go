package program

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Kind is the discriminator for Schema. The six values are the only types
// supported by the v0 DSL.
type Kind string

const (
	KindString Kind = "string"
	KindInt    Kind = "int"
	KindFloat  Kind = "float"
	KindBool   Kind = "bool"
	KindList   Kind = "list"
	KindObject Kind = "object"
)

// Schema describes the shape of a value in the GEPA-Go data model. Primitives
// carry only Kind; lists set Item; objects set Fields.
type Schema struct {
	Kind   Kind
	Item   *Schema
	Fields map[string]Schema
}

// UnmarshalJSON accepts either the bare-string form ("string", "list[int]",
// "list[list[bool]]") or the verbose object form ({"type":"object","fields":{...}}).
// Object form is required for object and list-of-object schemas.
func (s *Schema) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return fmt.Errorf("schema: empty value")
	}
	if trimmed[0] == '"' {
		var str string
		if err := json.Unmarshal(trimmed, &str); err != nil {
			return fmt.Errorf("schema: %w", err)
		}
		parsed, err := parseSchemaString(str)
		if err != nil {
			return err
		}
		*s = parsed
		return nil
	}
	if trimmed[0] == '{' {
		var aux struct {
			Type   string             `json:"type"`
			Item   *Schema            `json:"item,omitempty"`
			Fields map[string]Schema  `json:"fields,omitempty"`
			Extra  map[string]any     `json:"-"`
		}
		dec := json.NewDecoder(bytes.NewReader(trimmed))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&aux); err != nil {
			return fmt.Errorf("schema: %w", err)
		}
		return s.fromVerbose(aux.Type, aux.Item, aux.Fields)
	}
	return fmt.Errorf("schema: expected string or object, got %s", trimmed[:1])
}

func (s *Schema) fromVerbose(t string, item *Schema, fields map[string]Schema) error {
	switch Kind(t) {
	case KindString, KindInt, KindFloat, KindBool:
		if item != nil {
			return fmt.Errorf("schema: type=%q does not accept \"item\" field", t)
		}
		if fields != nil {
			return fmt.Errorf("schema: type=%q does not accept \"fields\" field", t)
		}
		*s = Schema{Kind: Kind(t)}
		return nil
	case KindList:
		if item == nil {
			return fmt.Errorf("schema: type=\"list\" requires \"item\" field")
		}
		if fields != nil {
			return fmt.Errorf("schema: type=\"list\" does not accept \"fields\" field")
		}
		*s = Schema{Kind: KindList, Item: item}
		return nil
	case KindObject:
		if item != nil {
			return fmt.Errorf("schema: type=\"object\" does not accept \"item\" field")
		}
		if fields == nil {
			return fmt.Errorf("schema: type=\"object\" requires \"fields\" field")
		}
		*s = Schema{Kind: KindObject, Fields: fields}
		return nil
	default:
		return fmt.Errorf("schema: unknown type %q", t)
	}
}

// parseSchemaString parses the bare-string form: primitives and list[T] recursive.
func parseSchemaString(str string) (Schema, error) {
	switch Kind(str) {
	case KindString, KindInt, KindFloat, KindBool:
		return Schema{Kind: Kind(str)}, nil
	}
	if strings.HasPrefix(str, "list[") && strings.HasSuffix(str, "]") {
		inner := str[len("list[") : len(str)-1]
		if inner == "" {
			return Schema{}, fmt.Errorf("schema: list[...] requires inner type, got %q", str)
		}
		innerSchema, err := parseSchemaString(inner)
		if err != nil {
			return Schema{}, err
		}
		return Schema{Kind: KindList, Item: &innerSchema}, nil
	}
	return Schema{}, fmt.Errorf("schema: unknown type %q", str)
}

// Validate checks that value (as produced by encoding/json into `any`) conforms
// to the Schema. Errors include the JSON path passed in by the caller; pass "" for
// no prefix or e.g. "input" to anchor messages.
func (s Schema) Validate(value any, path string) error {
	switch s.Kind {
	case KindString:
		if _, ok := value.(string); !ok {
			return fmt.Errorf("%s: expected string, got %s", path, goTypeName(value))
		}
		return nil
	case KindInt:
		f, ok := value.(float64)
		if !ok {
			return fmt.Errorf("%s: expected int, got %s", path, goTypeName(value))
		}
		if f != math.Trunc(f) || f < math.MinInt64 || f > math.MaxInt64 {
			return fmt.Errorf("%s: expected int, got non-integer %v", path, f)
		}
		return nil
	case KindFloat:
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("%s: expected float, got %s", path, goTypeName(value))
		}
		return nil
	case KindBool:
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%s: expected bool, got %s", path, goTypeName(value))
		}
		return nil
	case KindList:
		arr, ok := value.([]any)
		if !ok {
			return fmt.Errorf("%s: expected list, got %s", path, goTypeName(value))
		}
		for i, elem := range arr {
			if err := s.Item.Validate(elem, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
		return nil
	case KindObject:
		obj, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("%s: expected object, got %s", path, goTypeName(value))
		}
		for name, sub := range s.Fields {
			child, present := obj[name]
			if !present {
				return fmt.Errorf("%s: required field %q missing", path, name)
			}
			subPath := name
			if path != "" {
				subPath = path + "." + name
			}
			if err := sub.Validate(child, subPath); err != nil {
				return err
			}
		}
		for name := range obj {
			if _, declared := s.Fields[name]; !declared {
				return fmt.Errorf("%s: unknown field %q", path, name)
			}
		}
		return nil
	default:
		return fmt.Errorf("%s: schema has unknown kind %q", path, s.Kind)
	}
}

// ToJSONSchema emits a JSON Schema dict suitable for use as an OpenRouter
// `response_format` schema. Required in Phase 3; defined here since it lives
// next to the DSL.
func (s Schema) ToJSONSchema() map[string]any {
	switch s.Kind {
	case KindString:
		return map[string]any{"type": "string"}
	case KindInt:
		return map[string]any{"type": "integer"}
	case KindFloat:
		return map[string]any{"type": "number"}
	case KindBool:
		return map[string]any{"type": "boolean"}
	case KindList:
		return map[string]any{
			"type":  "array",
			"items": s.Item.ToJSONSchema(),
		}
	case KindObject:
		props := make(map[string]any, len(s.Fields))
		required := make([]string, 0, len(s.Fields))
		for name, sub := range s.Fields {
			props[name] = sub.ToJSONSchema()
			required = append(required, name)
		}
		sort.Strings(required)
		return map[string]any{
			"type":                 "object",
			"properties":           props,
			"required":             required,
			"additionalProperties": false,
		}
	}
	return map[string]any{}
}

// goTypeName names the Go runtime type produced by encoding/json for use in
// validation error messages.
func goTypeName(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "bool"
	case float64:
		return "float64"
	case string:
		return "string"
	case []any:
		return "list"
	case map[string]any:
		return "object"
	default:
		return fmt.Sprintf("%T", v)
	}
}
