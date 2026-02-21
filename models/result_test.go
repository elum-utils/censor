package models

import (
	"encoding/json"
	"testing"
)

func TestAIResultCompactUnmarshal(t *testing.T) {
	var r AIResult
	payload := []byte(`{"a":4,"b":"promo","c":0.91,"d":["buy now"],"e":77,"f":11}`)
	if err := json.Unmarshal(payload, &r); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	if r.StatusCode != StatusCommercialOffPlatform || r.ViolatorUserID != 77 || r.MessageID != 11 {
		t.Fatalf("unexpected result: %+v", r)
	}
}

func TestAIResultFullUnmarshal(t *testing.T) {
	var r AIResult
	payload := []byte(`{"status_code":5,"reason":"illegal","confidence":0.99,"trigger_tokens":["drug"]}`)
	if err := json.Unmarshal(payload, &r); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	if r.StatusCode != StatusDangerousIllegal {
		t.Fatalf("unexpected code: %d", r.StatusCode)
	}
}

func TestAIResultMarshalCompact(t *testing.T) {
	raw, err := json.Marshal(AIResult{StatusCode: StatusClean, Reason: "ok", Confidence: 1})
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	if string(raw) == "" || raw[0] != '{' {
		t.Fatalf("unexpected payload: %s", string(raw))
	}
}
