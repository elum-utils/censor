package models

import (
	"encoding/json"
	"fmt"
)

// StatusCode is a moderation decision code from AI.
type StatusCode int

const (
	StatusClean StatusCode = 1 + iota
	StatusNonCriticalAbuse
	StatusHumanReview
	StatusSuspicious
	StatusCommercialOffPlatform
	StatusDangerousIllegal
)

// Backward-compatible aliases (deprecated names).
const (
	StatusCritical = StatusDangerousIllegal
)

// Valid returns true when status code is in range [1..6].
func (s StatusCode) Valid() bool {
	return s >= StatusClean && s <= StatusCritical
}

// AIResult is a normalized AI response.
type AIResult struct {
	StatusCode     StatusCode `json:"status_code"`
	Reason         string     `json:"reason"`
	Confidence     float64    `json:"confidence"`
	TriggerTokens  []string   `json:"trigger_tokens"`
	ViolatorUserID int64      `json:"violator_user_id,omitempty"`
	MessageID      int64      `json:"message_id,omitempty"`
}

type aiResultAlias struct {
	StatusCode     StatusCode `json:"status_code"`
	Reason         string     `json:"reason"`
	Confidence     float64    `json:"confidence"`
	TriggerTokens  []string   `json:"trigger_tokens"`
	ViolatorUserID int64      `json:"violator_user_id,omitempty"`
	MessageID      int64      `json:"message_id,omitempty"`
}

type aiCompact struct {
	A StatusCode `json:"a"`
	B string     `json:"b"`
	C float64    `json:"c"`
	D []string   `json:"d"`
	E int64      `json:"e,omitempty"`
	F int64      `json:"f,omitempty"`
}

// UnmarshalJSON supports full and compact response formats.
func (r *AIResult) UnmarshalJSON(data []byte) error {
	var full aiResultAlias
	if err := json.Unmarshal(data, &full); err == nil && full.StatusCode != 0 {
		*r = AIResult(full)
		return nil
	}

	var compact aiCompact
	if err := json.Unmarshal(data, &compact); err == nil && compact.A != 0 {
		r.StatusCode = compact.A
		r.Reason = compact.B
		r.Confidence = compact.C
		r.TriggerTokens = compact.D
		r.ViolatorUserID = compact.E
		r.MessageID = compact.F
		return nil
	}

	return fmt.Errorf("models: unsupported AI result format")
}

// MarshalJSON emits compact format for payload size efficiency.
func (r AIResult) MarshalJSON() ([]byte, error) {
	return json.Marshal(aiCompact{
		A: r.StatusCode,
		B: r.Reason,
		C: r.Confidence,
		D: r.TriggerTokens,
		E: r.ViolatorUserID,
		F: r.MessageID,
	})
}

// Violation describes a final moderation decision for one message.
type Violation struct {
	Message   Message
	AIResult  AIResult
	Triggered bool
}
