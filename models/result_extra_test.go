package models

import (
	"encoding/json"
	"testing"
)

func TestStatusValid(t *testing.T) {
	if !StatusClean.Valid() || !StatusCritical.Valid() {
		t.Fatalf("valid statuses expected")
	}
	if StatusCode(0).Valid() || StatusCode(7).Valid() {
		t.Fatalf("invalid statuses expected")
	}
}

func TestAIResultUnmarshalInvalid(t *testing.T) {
	var r AIResult
	if err := json.Unmarshal([]byte(`{"x":1}`), &r); err == nil {
		t.Fatalf("expected error")
	}
}
