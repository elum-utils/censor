package ai

import (
	"testing"

	"github.com/elum-utils/censor/models"
)

func TestParseResultsArrayCompact(t *testing.T) {
	results, err := parseResults(`[{"a":2,"b":"abuse","c":0.8,"d":["bad"],"e":1,"f":10}]`)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(results) != 1 || results[0].StatusCode != models.StatusNonCriticalAbuse {
		t.Fatalf("unexpected parse result: %+v", results)
	}
}

func TestAlignResultsByMessageID(t *testing.T) {
	msgs := []models.Message{{ID: 10, User: 2}, {ID: 11, User: 3}}
	in := []models.AIResult{{MessageID: 11, StatusCode: models.StatusCritical}, {MessageID: 10, StatusCode: models.StatusClean}}
	out := alignResults(msgs, in)
	if len(out) != 2 || out[0].MessageID != 10 || out[1].MessageID != 11 {
		t.Fatalf("unexpected align: %+v", out)
	}
}
