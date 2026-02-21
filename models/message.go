package models

// Message is an input unit for moderation.
type Message struct {
	ID       int64  `json:"id"`
	DialogID string `json:"dialog_id,omitempty"`
	User     int64  `json:"user"`
	Data     string `json:"data"`
}
