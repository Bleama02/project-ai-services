package openapi

import (
	"encoding/json"

	"github.com/google/jsonschema-go/jsonschema"
	base "github.com/pb33f/libopenapi/datamodel/high/base"
	"gopkg.in/yaml.v3"
)

// ConvertSchemaToJSONSchema converts a libopenapi schema to a JSON Schema
func ConvertSchemaToJSONSchema(schema *base.SchemaProxy) *jsonschema.Schema {
	if schema == nil {
		return nil
	}

	// Build the schema
	built := schema.Schema()
	if built == nil {
		return nil
	}

	// Render the schema to YAML bytes and convert to map
	renderedYAML, _ := built.Render()
	if renderedYAML == nil {
		return nil
	}

	// Parse YAML to a map
	var schemaMap map[string]interface{}
	if err := yaml.Unmarshal(renderedYAML, &schemaMap); err != nil {
		return nil
	}

	// Marshal to JSON and unmarshal to jsonschema.Schema
	jsonData, err := json.Marshal(schemaMap)
	if err != nil {
		return nil
	}

	var js *jsonschema.Schema
	if err := json.Unmarshal(jsonData, &js); err != nil {
		return nil
	}

	restoreContentMediaType(built, js)

	return js
}

// restoreContentMediaType copies contentMediaType from the low-level model onto top-level
// properties and their array items. libopenapi's high-level Schema has no such field, so
// Render() drops it, but OpenAPI 3.1 (e.g. FastAPI) uses it to mark multipart file fields.
func restoreContentMediaType(built *base.Schema, js *jsonschema.Schema) {
	if built == nil || js == nil || built.Properties == nil {
		return
	}
	for pair := built.Properties.First(); pair != nil; pair = pair.Next() {
		target := js.Properties[pair.Key()]
		prop := pair.Value().Schema()
		if target == nil || prop == nil {
			continue
		}
		if low := prop.GoLow(); low != nil && target.ContentMediaType == "" {
			target.ContentMediaType = low.ContentMediaType.Value
		}
		if prop.Items != nil && prop.Items.IsA() && target.Items != nil {
			if item := prop.Items.A.Schema(); item != nil {
				if low := item.GoLow(); low != nil && target.Items.ContentMediaType == "" {
					target.Items.ContentMediaType = low.ContentMediaType.Value
				}
			}
		}
	}
}
