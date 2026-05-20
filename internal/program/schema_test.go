package program

import (
	"encoding/json"
	"reflect"
	"testing"
)

func unmarshalSchema(t *testing.T, src string) Schema {
	t.Helper()
	var s Schema
	if err := json.Unmarshal([]byte(src), &s); err != nil {
		t.Fatalf("unmarshal %s: %v", src, err)
	}
	return s
}

func TestSchemaParseStringForm(t *testing.T) {
	cases := []struct {
		src  string
		kind Kind
	}{
		{`"string"`, KindString},
		{`"int"`, KindInt},
		{`"float"`, KindFloat},
		{`"bool"`, KindBool},
	}
	for _, c := range cases {
		s := unmarshalSchema(t, c.src)
		if s.Kind != c.kind {
			t.Errorf("%s: kind = %q, want %q", c.src, s.Kind, c.kind)
		}
		if s.Item != nil || s.Fields != nil {
			t.Errorf("%s: primitive should have no Item/Fields", c.src)
		}
	}
}

func TestSchemaParseListString(t *testing.T) {
	s := unmarshalSchema(t, `"list[int]"`)
	if s.Kind != KindList || s.Item == nil || s.Item.Kind != KindInt {
		t.Errorf("list[int] parsed wrong: %+v", s)
	}
}

func TestSchemaParseNestedList(t *testing.T) {
	s := unmarshalSchema(t, `"list[list[bool]]"`)
	if s.Kind != KindList || s.Item.Kind != KindList || s.Item.Item.Kind != KindBool {
		t.Errorf("list[list[bool]] parsed wrong: %+v", s)
	}
}

func TestSchemaParseVerboseObject(t *testing.T) {
	src := `{"type":"object","fields":{"q":"string","n":"int"}}`
	s := unmarshalSchema(t, src)
	if s.Kind != KindObject || len(s.Fields) != 2 {
		t.Fatalf("got %+v", s)
	}
	if s.Fields["q"].Kind != KindString || s.Fields["n"].Kind != KindInt {
		t.Errorf("field types wrong: %+v", s.Fields)
	}
}

func TestSchemaParseListOfObjects(t *testing.T) {
	src := `{"type":"list","item":{"type":"object","fields":{"text":"string"}}}`
	s := unmarshalSchema(t, src)
	if s.Kind != KindList || s.Item.Kind != KindObject {
		t.Fatalf("got %+v", s)
	}
	if s.Item.Fields["text"].Kind != KindString {
		t.Errorf("nested object field wrong: %+v", s.Item.Fields)
	}
}

func TestSchemaRejectUnknownPrimitive(t *testing.T) {
	var s Schema
	err := json.Unmarshal([]byte(`"strng"`), &s)
	if err == nil {
		t.Fatal("expected error")
	}
	want := `schema: unknown type "strng"`
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestSchemaRejectUnknownVerboseType(t *testing.T) {
	var s Schema
	err := json.Unmarshal([]byte(`{"type":"bytes"}`), &s)
	if err == nil {
		t.Fatal("expected error")
	}
	want := `schema: unknown type "bytes"`
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestSchemaRejectListWithoutItem(t *testing.T) {
	var s Schema
	err := json.Unmarshal([]byte(`{"type":"list"}`), &s)
	if err == nil || err.Error() != `schema: type="list" requires "item" field` {
		t.Errorf("got %v", err)
	}
}

func TestSchemaRejectObjectWithoutFields(t *testing.T) {
	var s Schema
	err := json.Unmarshal([]byte(`{"type":"object"}`), &s)
	if err == nil || err.Error() != `schema: type="object" requires "fields" field` {
		t.Errorf("got %v", err)
	}
}

func TestSchemaRejectListWithFields(t *testing.T) {
	var s Schema
	err := json.Unmarshal([]byte(`{"type":"list","item":"int","fields":{"x":"string"}}`), &s)
	if err == nil || err.Error() != `schema: type="list" does not accept "fields" field` {
		t.Errorf("got %v", err)
	}
}

func TestSchemaRejectEmptyListBrackets(t *testing.T) {
	var s Schema
	err := json.Unmarshal([]byte(`"list[]"`), &s)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestValidatePrimitives(t *testing.T) {
	cases := []struct {
		schema string
		value  any
		ok     bool
		err    string
	}{
		{`"string"`, "hi", true, ""},
		{`"string"`, 1.0, false, `x: expected string, got float64`},
		{`"int"`, 3.0, true, ""},
		{`"int"`, 3.5, false, `x: expected int, got non-integer 3.5`},
		{`"int"`, "3", false, `x: expected int, got string`},
		{`"float"`, 1.5, true, ""},
		{`"float"`, true, false, `x: expected float, got bool`},
		{`"bool"`, true, true, ""},
		{`"bool"`, "yes", false, `x: expected bool, got string`},
	}
	for _, c := range cases {
		s := unmarshalSchema(t, c.schema)
		err := s.Validate(c.value, "x")
		if c.ok {
			if err != nil {
				t.Errorf("%s %v: unexpected error %v", c.schema, c.value, err)
			}
			continue
		}
		if err == nil {
			t.Errorf("%s %v: expected error", c.schema, c.value)
			continue
		}
		if err.Error() != c.err {
			t.Errorf("%s %v: got %q, want %q", c.schema, c.value, err.Error(), c.err)
		}
	}
}

func TestValidateListOfInt(t *testing.T) {
	s := unmarshalSchema(t, `"list[int]"`)
	if err := s.Validate([]any{1.0, 2.0, 3.0}, "xs"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	err := s.Validate([]any{1.0, "two", 3.0}, "xs")
	if err == nil || err.Error() != `xs[1]: expected int, got string` {
		t.Errorf("got %v", err)
	}
}

func TestValidateObjectRequiredField(t *testing.T) {
	s := unmarshalSchema(t, `{"type":"object","fields":{"a":"string","b":"int"}}`)
	err := s.Validate(map[string]any{"a": "x"}, "input")
	if err == nil || err.Error() != `input: required field "b" missing` {
		t.Errorf("got %v", err)
	}
}

func TestValidateObjectStrictNoExtras(t *testing.T) {
	s := unmarshalSchema(t, `{"type":"object","fields":{"a":"string"}}`)
	err := s.Validate(map[string]any{"a": "x", "extra": "y"}, "input")
	if err == nil || err.Error() != `input: unknown field "extra"` {
		t.Errorf("got %v", err)
	}
}

func TestValidateNestedObjectPath(t *testing.T) {
	s := unmarshalSchema(t, `{"type":"object","fields":{"outer":{"type":"object","fields":{"inner":"string"}}}}`)
	err := s.Validate(map[string]any{"outer": map[string]any{"inner": 1.0}}, "input")
	if err == nil || err.Error() != `input.outer.inner: expected string, got float64` {
		t.Errorf("got %v", err)
	}
}

func TestToJSONSchemaPrimitive(t *testing.T) {
	s := unmarshalSchema(t, `"string"`)
	got := s.ToJSONSchema()
	want := map[string]any{"type": "string"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestToJSONSchemaObject(t *testing.T) {
	s := unmarshalSchema(t, `{"type":"object","fields":{"q":"string","n":"int"}}`)
	got := s.ToJSONSchema()
	want := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"q": map[string]any{"type": "string"},
			"n": map[string]any{"type": "integer"},
		},
		"required":             []string{"n", "q"},
		"additionalProperties": false,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v\nwant %v", got, want)
	}
}

func TestToJSONSchemaListOfObjects(t *testing.T) {
	s := unmarshalSchema(t, `{"type":"list","item":{"type":"object","fields":{"text":"string"}}}`)
	got := s.ToJSONSchema()
	want := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"text": map[string]any{"type": "string"}},
			"required":             []string{"text"},
			"additionalProperties": false,
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v\nwant %v", got, want)
	}
}
