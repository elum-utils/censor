package engine

import "testing"

func TestEngineAddRemoveBranchesAndStats(t *testing.T) {
	e := New()
	if e.AddToken(" ") {
		t.Fatalf("empty token must be ignored")
	}
	if !e.AddToken("hello world") {
		t.Fatalf("token must be added")
	}
	if e.AddToken("hello world") {
		t.Fatalf("duplicate token should be ignored")
	}
	if !e.RemoveToken("hello world") {
		t.Fatalf("token must be removed")
	}
	if e.RemoveToken("hello world") {
		t.Fatalf("missing token should not be removed")
	}
	_ = e.FindTriggers("x")
	st := e.Stats()
	if st.TotalLookups == 0 {
		t.Fatalf("expected lookups metric")
	}
}

func TestFindTriggersEmptyAndNoToken(t *testing.T) {
	e := New()
	if got := e.FindTriggers(""); got != nil {
		t.Fatalf("expected nil")
	}
	e.AddToken("abc")
	if got := e.FindTriggers("zzz"); got != nil {
		t.Fatalf("expected nil")
	}
}
